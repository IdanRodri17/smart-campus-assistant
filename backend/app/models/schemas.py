"""
Pydantic models for API request/response schemas.
Defines the contract between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# ── Enums ──


class QuestionCategory(str, Enum):
    """Categories for question classification."""

    SCHEDULE = "schedule"
    GENERAL_INFO = "general_info"
    TECHNICAL_ISSUE = "technical_issue"
    OUT_OF_SCOPE = "out_of_scope"


# ── Request Models ──


class HistoryMessage(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., max_length=1000)


class AskRequest(BaseModel):
    """Request body for POST /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The student's natural-language question",
        examples=["What are the office hours for Dr. Cohen?"],
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Previous turns in the conversation (max 10 messages = 5 turns)",
    )


class RatingRequest(BaseModel):
    """Request body for POST /ask endpoint"""

    interaction_id: str
    rating: bool = Field(..., description="True = thumbs up, False = thumbs down")
    feedback: Optional[str] = Field(None, max_length=500)


# ── Response Models ──


class AskResponse(BaseModel):
    """Successful response from the AI assistant."""

    answer: str
    category: QuestionCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    response_time_ms: int
    interaction_id: str


class FallbackResponse(BaseModel):
    """Response when AI confidence is below threshold."""

    message: str = (
        "I'm not confident enough to give you an accurate answer for this question."
    )
    suggestion: str = "Please contact campus support for assistance."
    staff_contact: dict = Field(
        default_factory=lambda: {
            "email": "support@campus.ac.il",
            "phone": "03-123-4567",
            "office": "Building A, Room 101",
        }
    )
    category: QuestionCategory
    interaction_id: str


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    database: bool
    ai_service: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    status_code: int


# ── Admin Models ──


class CampusRecordCreate(BaseModel):
    """Request body for creating a campus data record."""

    category: str = Field(..., description="schedule | general_info | technical_issue")
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


class CampusRecordUpdate(BaseModel):
    """Request body for updating a campus data record (all fields optional)."""

    category: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
