"""
Question classifier — categorizes student questions into predefined categories.
Uses a lightweight LLM call with structured output for reliable classification.
"""

import json
import logging
from openai import OpenAI
from app.core.config import get_settings
from app.core.cache import cache_get, cache_set, CLASSIFICATION_TTL
from app.models.schemas import QuestionCategory

logger = logging.getLogger(__name__)

# Module-level singleton — created once, reused for every request
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key)
    return _client


# ── Keyword pre-classifier ──────────────────────────────────────────────────
# Fast local check before spending an LLM call. If exactly ONE category has
# a keyword hit, return it immediately. Zero or multiple hits → fall through
# to the LLM for proper disambiguation.

_KEYWORD_MAP: dict[QuestionCategory, list[str]] = {
    QuestionCategory.SCHEDULE: [
        "office hours", "שעות קבלה", "exam", "בחינה", "מבחן", "test date",
        "schedule", "timetable", "deadline", "submission", "due date",
        "when is", "appointment", "class time", "lecture", "semester",
        "professor available", "teacher available",
    ],
    QuestionCategory.GENERAL_INFO: [
        "where is", "cafeteria", "library", "parking", "registration",
        "how do i register", "campus map", "building", "facility",
        "contact", "phone number", "email address", "open hours",
        "student card", "tuition", "scholarship", "policy",
    ],
    QuestionCategory.TECHNICAL_ISSUE: [
        "wifi", "wi-fi", "password", "can't login", "cannot login",
        "portal", "student system", "vpn", "software", "lab computer",
        "printing", "print", "it support", "not working", "broken",
        "error", "access denied", "reset password", "account locked",
    ],
    QuestionCategory.OUT_OF_SCOPE: [
        "tell me a joke", "what is the meaning of life", "who is the president",
        "weather", "stock price", "recipe", "movie", "sports score",
    ],
}


def _keyword_classify(question: str) -> QuestionCategory | None:
    """
    Attempt fast keyword-based classification.
    Returns a category only when exactly one category matches — avoids
    mis-classifying ambiguous questions that mention multiple topics.
    """
    lower = question.lower()
    matches = [
        cat
        for cat, keywords in _KEYWORD_MAP.items()
        if any(kw in lower for kw in keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous or no match → let the LLM decide


# ── LLM classification prompt ────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are a question classifier for a university campus assistant.

Classify the following student question into EXACTLY ONE category:
- "schedule": Questions about office hours (שעות קבלה), exam dates, class times, room availability, timetables, when/where a professor is available, appointment scheduling, submission deadlines
- "general_info": Questions about campus facilities, services, policies, FAQs, locations, registration, parking, cafeteria, library
- "technical_issue": Questions about IT problems, WiFi, student portal, printing, software access, VPN, lab computers
- "out_of_scope": Questions unrelated to campus (jokes, personal advice, general knowledge, politics)

Examples:
- "What are the office hours for Dr. Cohen?" → "schedule"
- "When is the Python exam?" → "schedule"  
- "Where is the cafeteria?" → "general_info"
- "I can't log into the portal" → "technical_issue"
- "Tell me a joke" → "out_of_scope"

Respond with ONLY a JSON object:
{{"category": "<category>", "reasoning": "<one sentence why>"}}

Student question: "{question}"
"""


def classify_question(question: str) -> tuple[QuestionCategory, str]:
    """
    Classify a student question into a category.

    Args:
        question: The student's question text.

    Returns:
        Tuple of (category, reasoning).
    """
    # ── 1. Fast keyword pre-classifier (no API call) ──
    keyword_result = _keyword_classify(question)
    if keyword_result is not None:
        logger.info(f"Keyword-classified question as '{keyword_result.value}'")
        return keyword_result, "keyword match"

    # ── 2. Redis cache (avoid repeat LLM calls for identical questions) ──
    cached = cache_get("cls", question)
    if cached is not None:
        logger.debug("Classification cache hit")
        return QuestionCategory(cached["category"]), cached["reasoning"]

    # ── 3. LLM classification (only reached for ambiguous/unseen questions) ──
    client = _get_client()
    settings = get_settings()

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": CLASSIFICATION_PROMPT.format(question=question),
                },
            ],
            temperature=0.0,  # Deterministic classification
            max_tokens=100,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content.strip()

        result = json.loads(result_text)
        category = QuestionCategory(result["category"])
        reasoning = result.get("reasoning", "")

        cache_set("cls", question, {"category": category.value, "reasoning": reasoning}, CLASSIFICATION_TTL)
        logger.info(f"LLM-classified question as '{category.value}': {reasoning}")
        return category, reasoning

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(
            f"Classification parsing failed: {e}. Defaulting to general_info."
        )
        return (
            QuestionCategory.GENERAL_INFO,
            "Classification failed — defaulting to general_info",
        )

    except Exception as e:
        logger.error(f"Classification API error: {e}")
        return QuestionCategory.GENERAL_INFO, f"API error — defaulting to general_info"
