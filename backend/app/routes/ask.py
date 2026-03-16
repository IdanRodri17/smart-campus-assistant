"""
API routes for the Smart Campus Assistant.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import (
    AskRequest,
    AskResponse,
    FallbackResponse,
    HealthResponse,
    ErrorResponse,
    RatingRequest,
)
from sqlalchemy import text
from app.ai.orchestrator import process_question
from app.core.database import check_db_health, get_db_engine
from app.core.config import get_settings
from app.core.limiter import limiter, RATE_LIMIT

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
@limiter.limit(RATE_LIMIT)
async def ask_question(request: Request, body: AskRequest):
    """
    Main endpoint — processes a student question through the full AI pipeline.

    Pipeline: validate → classify → RAG retrieve → LLM generate → confidence check → respond
    """
    settings = get_settings()

    # Input validation (beyond Pydantic's min/max length)
    question = body.question.strip()
    if len(question) < 3:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid question with at least 3 characters.",
        )

    logger.info(f"Received question: '{question[:80]}...'")

    try:
        history = [msg.model_dump() for msg in body.history]
        response = process_question(question, history=history)
        return response

    except Exception as e:
        logger.error(f"Unexpected error processing question: {e}")
        raise HTTPException(
            status_code=503,
            detail="Our AI service is temporarily unavailable. Please try again shortly.",
        )


@router.post(
    "/feedback",
    summary="Submit answer feedback",
    description="Rate an answer as helpful (thumbs up) or not helpful (thumbs down).",
)
@limiter.limit("60/minute")
async def submit_feedback(request: Request, body: RatingRequest):
    """Store a thumbs-up / thumbs-down rating for a previous interaction."""
    try:
        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ratings (interaction_id, rating, feedback)
                    VALUES (:interaction_id, :rating, :feedback)
                    """
                ),
                {
                    "interaction_id": body.interaction_id,
                    "rating": body.rating,
                    "feedback": body.feedback,
                },
            )
        logger.info(f"Feedback recorded for interaction {body.interaction_id}: {'👍' if body.rating else '👎'}")
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=503, detail="Could not save feedback.")


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
