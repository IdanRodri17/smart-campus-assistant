"""
AI Orchestrator — the main pipeline that processes a student question end-to-end.
Ties together: input validation → classification → RAG retrieval → LLM generation → fallback.
"""

import logging
import time
import uuid
from app.ai.classifier import classify_question
from app.ai.rag import retrieve_context, get_structured_data
from app.ai.generator import generate_answer
from app.core.config import get_settings
from app.models.schemas import QuestionCategory, AskResponse, FallbackResponse

logger = logging.getLogger(__name__)

# Common prompt injection patterns to flag
INJECTION_PATTERNS = [
    "ignore previous",
    "ignore above",
    "ignore all",
    "you are now",
    "new instructions",
    "system prompt",
    "forget your",
    "disregard",
    "override",
]


def sanitize_input(question: str) -> tuple[str, bool]:
    """
    Sanitize user input and detect potential prompt injection.

    Args:
        question: Raw user input.

    Returns:
        Tuple of (sanitized_question, is_suspicious).
    """
    cleaned = question.strip()
    lower = cleaned.lower()

    is_suspicious = any(pattern in lower for pattern in INJECTION_PATTERNS)

    if is_suspicious:
        logger.warning(f"Potential prompt injection detected: '{cleaned[:50]}...'")

    return cleaned, is_suspicious


def process_question(question: str) -> AskResponse | FallbackResponse:
    """
    Main AI pipeline — processes a student question end-to-end.

    Pipeline:
    1. Sanitize input
    2. Classify question category
    3. Retrieve context (RAG + structured data in parallel)
    4. Generate answer with LLM
    5. Apply confidence threshold (answer or fallback)
    6. Return response

    Args:
        question: The student's question.

    Returns:
        AskResponse (if confident) or FallbackResponse (if not).
    """
    settings = get_settings()
    start_time = time.time()
    interaction_id = str(uuid.uuid4())

    # Step 1: Sanitize
    cleaned_question, is_suspicious = sanitize_input(question)

    # Step 2: Classify
    category, classification_reason = classify_question(cleaned_question)
    logger.info(
        f"[{interaction_id}] Category: {category.value} — {classification_reason}"
    )

    # Early exit for out-of-scope
    if category == QuestionCategory.OUT_OF_SCOPE:
        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[{interaction_id}] Out-of-scope question, returning fallback")
        return FallbackResponse(
            message="This question doesn't seem to be about campus information. I can only help with campus-related questions.",
            suggestion="Try asking about office hours, exam schedules, room locations, or technical support.",
            category=category,
            interaction_id=interaction_id,
        )

    # Step 3: Retrieve context (RAG with metadata filtering + structured data)
    rag_chunks = retrieve_context(cleaned_question, category=category.value)
    structured_data = get_structured_data(category.value)

    logger.info(
        f"[{interaction_id}] Retrieved {len(rag_chunks)} RAG chunks, "
        f"{len(structured_data)} structured records"
    )

    # Step 4: Generate answer
    try:
        result = generate_answer(
            question=cleaned_question,
            category=category,
            rag_chunks=rag_chunks,
            structured_data=structured_data,
        )
    except Exception as e:
        logger.error(f"[{interaction_id}] LLM generation failed: {e}")
        return FallbackResponse(
            message="Our AI service is temporarily unavailable. Your question has been saved.",
            suggestion="Please try again in a few minutes, or contact campus support directly.",
            category=category,
            interaction_id=interaction_id,
        )

    total_time_ms = int((time.time() - start_time) * 1000)

    # Step 5: Confidence threshold check
    if result["confidence"] < settings.confidence_threshold:
        logger.info(
            f"[{interaction_id}] Low confidence ({result['confidence']:.2f}), returning fallback"
        )
        return FallbackResponse(
            message=f"I found some information but I'm not confident enough to give you an accurate answer (confidence: {result['confidence']:.0%}).",
            suggestion="Please contact campus support for a reliable answer to this question.",
            category=category,
            interaction_id=interaction_id,
        )

    # Step 6: Return confident answer
    logger.info(
        f"[{interaction_id}] Success: confidence={result['confidence']:.2f}, "
        f"time={total_time_ms}ms, tokens={result['tokens_used']}"
    )

    return AskResponse(
        answer=result["answer"],
        category=category,
        confidence=result["confidence"],
        sources=result["sources_used"],
        response_time_ms=total_time_ms,
        interaction_id=interaction_id,
    )
