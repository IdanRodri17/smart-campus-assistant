"""
Question classifier — categorizes student questions into predefined categories.
Uses a lightweight LLM call with structured output for reliable classification.
"""

import json
import logging
from openai import OpenAI
from app.core.config import get_settings
from app.models.schemas import QuestionCategory

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a question classifier for a university campus assistant.

Classify the following student question into EXACTLY ONE category:
- "schedule": Questions about office hours, exam dates, class times, room availability, timetables
- "general_info": Questions about campus facilities, services, policies, FAQs, locations, registration
- "technical_issue": Questions about IT problems, WiFi, student portal, printing, software access
- "out_of_scope": Questions unrelated to campus (jokes, personal advice, general knowledge)

Respond with ONLY a JSON object:
{"category": "<category>", "reasoning": "<one sentence why>"}

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
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

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
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        result = json.loads(result_text)
        category = QuestionCategory(result["category"])
        reasoning = result.get("reasoning", "")

        logger.info(f"Classified question as '{category.value}': {reasoning}")
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
