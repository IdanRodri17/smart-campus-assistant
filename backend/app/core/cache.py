"""
Redis cache layer for Smart Campus Assistant.
Caches embeddings and classifications to reduce OpenAI API calls.

Degrades gracefully — if Redis is unavailable (no REDIS_URL set, or connection
fails) every cache call is a no-op and the system works without caching.
"""

import hashlib
import json
import logging
from typing import Any

import redis
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# TTLs
EMBEDDING_TTL = 86_400       # 24 hours  — same text → same vector
CLASSIFICATION_TTL = 604_800  # 7 days   — question meaning rarely changes

_client: redis.Redis | None = None
_client_initialized = False


def _get_client() -> redis.Redis | None:
    """Return a live Redis client, or None if Redis is not configured/reachable."""
    global _client, _client_initialized
    if _client_initialized:
        return _client

    _client_initialized = True
    settings = get_settings()

    if not settings.redis_url:
        logger.info("REDIS_URL not set — caching disabled")
        return None

    try:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
        _client.ping()
        logger.info("Redis connected — caching enabled")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — caching disabled")
        _client = None

    return _client


def _make_key(prefix: str, text: str) -> str:
    """SHA-256 hash of the text to keep keys short and safe."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"campus:{prefix}:{digest}"


def cache_get(prefix: str, text: str) -> Any | None:
    """Return cached value, or None on miss / Redis unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(_make_key(prefix, text))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.debug(f"Cache GET error: {e}")
        return None


def cache_set(prefix: str, text: str, value: Any, ttl: int) -> None:
    """Store a value. Silently ignores errors."""
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(_make_key(prefix, text), ttl, json.dumps(value))
    except Exception as e:
        logger.debug(f"Cache SET error: {e}")
