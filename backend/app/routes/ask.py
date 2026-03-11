"""
API routes for the Smart Campus Assistant.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    AskRequest,
    AskResponse,
    FallbackResponse,
    HealthResponse,
    ErrorResponse,
)
from app.ai.orchestrator import process_question
from app.core.database import check_db_health
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse | FallbackResponse,
    responses={
        200: {"description": "AI-generated answer or fallback message"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
    },
    summary="Ask a campus question",
    description="Submit a natural-language question about campus information and receive an AI-generated answer.",
)
async def ask_question(request: AskRequest):
    """
    Main endpoint — processes a student question through the full AI pipeline.

    Pipeline: validate → classify → RAG retrieve → LLM generate → confidence check → respond
    """
    settings = get_settings()

    # Input validation (beyond Pydantic's min/max length)
    question = request.question.strip()
    if len(question) < 3:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid question with at least 3 characters.",
        )

    logger.info(f"Received question: '{question[:80]}...'")

    try:
        response = process_question(question)
        return response

    except Exception as e:
        logger.error(f"Unexpected error processing question: {e}")
        raise HTTPException(
            status_code=503,
            detail="Our AI service is temporarily unavailable. Please try again shortly.",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Check the health of the API, database connection, and AI service availability.",
)
async def health_check():
    """Health check endpoint for monitoring and Docker healthchecks."""
    db_healthy = check_db_health()

    # Simple AI check — verify OpenAI key is configured
    settings = get_settings()
    ai_healthy = bool(settings.openai_api_key)

    overall = "healthy" if (db_healthy and ai_healthy) else "degraded"

    return HealthResponse(
        status=overall,
        database=db_healthy,
        ai_service=ai_healthy,
        timestamp=datetime.now(timezone.utc),
    )
