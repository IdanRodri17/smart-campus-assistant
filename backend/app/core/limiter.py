"""
Rate limiter for Smart Campus Assistant.

Uses slowapi (wrapper around the `limits` library).
Storage backend: Redis when REDIS_URL is configured, in-memory otherwise.
Limit is driven by the RATE_LIMIT_PER_MINUTE config value (default: 30).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import get_settings

_settings = get_settings()

# Use Redis as the shared counter store when available so rate limits
# are enforced correctly across multiple worker processes.
_storage_uri = _settings.redis_url if _settings.redis_url else "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_storage_uri,
)

# Pre-built limit string consumed by route decorators, e.g. "30/minute"
RATE_LIMIT = f"{_settings.rate_limit_per_minute}/minute"
