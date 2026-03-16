"""
AI Orchestrator — the main pipeline that processes a student question end-to-end.
Ties together: input validation → classification → RAG retrieval → LLM generation → fallback → logging.
"""

import logging
import time
import uuid
from sqlalchemy import text
from app.ai.classifier import classify_question
from app.ai.rag import retrieve_context, get_structured_data
from app.ai.generator import generate_answer
from app.core.config import get_settings
from app.core.database import get_db_engine
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


def log_interaction(
    interaction_id: str,
    question: str,
    answer: str,
    category: str,
    confidence: float,
    response_time_ms: int,
    tokens_used: int = 0,
    prompt_version: str = "",
    is_fallback: bool = False,
):
    """
    Log the interaction to the database for analytics and experiments tracking (FR-07).
    Runs in a try/except so logging failures never break the user response.
    """
    try:
        engine = get_db_engine()
        query = text(
            """
            INSERT INTO interactions (id, question, answer, category, confidence, 
                                      response_time_ms, tokens_used, prompt_version, is_fallback)
            VALUES (:id, :question, :answer, :category, :confidence, 
                    :response_time_ms, :tokens_used, :prompt_version, :is_fallback)
            """
        )
        with engine.begin() as conn:
            conn.execute(
                query,
                {
                    "id": interaction_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "confidence": confidence,
                    "response_time_ms": response_time_ms,
                    "tokens_used": tokens_used,
                    "prompt_version": prompt_version,
                    "is_fallback": is_fallback,
                },
            )
        logger.info(f"[{interaction_id}] Interaction logged to database")
    except Exception as e:
        logger.error(f"[{interaction_id}] Failed to log interaction: {e}")


def process_question(question: str) -> AskResponse | FallbackResponse:
    """
    Main AI pipeline — processes a student question end-to-end.

    Pipeline:
    1. Sanitize input
    2. Classify question category
    3. Retrieve context (RAG with metadata filtering + structured data)
    4. Generate answer with LLM
    5. Apply confidence threshold (answer or fallback)
    6. Log interaction to database
    7. Return response
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

        fallback = FallbackResponse(
            message="This question doesn't seem to be about campus information. I can only help with campus-related questions.",
            suggestion="Try asking about office hours, exam schedules, room locations, or technical support.",
            category=category,
            interaction_id=interaction_id,
        )

        # Log out-of-scope interaction
        log_interaction(
            interaction_id=interaction_id,
            question=cleaned_question,
            answer=fallback.message,
            category=category.value,
            confidence=0.0,
            response_time_ms=total_time_ms,
            is_fallback=True,
        )

        return fallback

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
        fallback = FallbackResponse(
            message="Our AI service is temporarily unavailable. Your question has been saved.",
            suggestion="Please try again in a few minutes, or contact campus support directly.",
            category=category,
            interaction_id=interaction_id,
        )

        log_interaction(
            interaction_id=interaction_id,
            question=cleaned_question,
            answer=fallback.message,
            category=category.value,
            confidence=0.0,
            response_time_ms=int((time.time() - start_time) * 1000),
            is_fallback=True,
        )

        return fallback

    total_time_ms = int((time.time() - start_time) * 1000)

    # Step 5: Confidence threshold check
    if result["confidence"] < settings.confidence_threshold:
        logger.info(
            f"[{interaction_id}] Low confidence ({result['confidence']:.2f}), returning fallback"
        )

        fallback = FallbackResponse(
            message=f"I found some information but I'm not confident enough to give you an accurate answer (confidence: {result['confidence']:.0%}).",
            suggestion="Please contact campus support for a reliable answer to this question.",
            category=category,
            interaction_id=interaction_id,
        )

        log_interaction(
            interaction_id=interaction_id,
            question=cleaned_question,
            answer=fallback.message,
            category=category.value,
            confidence=result["confidence"],
            response_time_ms=total_time_ms,
            tokens_used=result.get("tokens_used", 0),
            prompt_version=result.get("prompt_version", ""),
            is_fallback=True,
        )

        return fallback

    # Step 6: Return confident answer + log
    logger.info(
        f"[{interaction_id}] Success: confidence={result['confidence']:.2f}, "
        f"time={total_time_ms}ms, tokens={result['tokens_used']}"
    )

    log_interaction(
        interaction_id=interaction_id,
        question=cleaned_question,
        answer=result["answer"],
        category=category.value,
        confidence=result["confidence"],
        response_time_ms=total_time_ms,
        tokens_used=result.get("tokens_used", 0),
        prompt_version=result.get("prompt_version", ""),
        is_fallback=False,
    )

    return AskResponse(
        answer=result["answer"],
        category=category,
        confidence=result["confidence"],
        sources=result["sources_used"],
        response_time_ms=total_time_ms,
        interaction_id=interaction_id,
    )
