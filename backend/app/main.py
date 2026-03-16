"""
Smart Campus Assistant — FastAPI Application
Main entry point for the backend server.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded

from app.routes.ask import router as ask_router
from app.routes.admin import router as admin_router
from app.core.config import get_settings
from app.core.limiter import limiter

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ──
settings = get_settings()

async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return a JSON 429 instead of slowapi's plain-text default."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"Rate limit exceeded — max {settings.rate_limit_per_minute} "
                "requests per minute. Please wait and try again."
            )
        },
    )


app = FastAPI(
    title="Smart Campus Assistant API",
    description=(
        "AI-powered campus information assistant. "
        "Ask questions about office hours, exam schedules, rooms, and more."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate limiter ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics ──
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Routes ──
app.include_router(ask_router, prefix="/api", tags=["Campus Assistant"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API info."""
    return {
        "service": "Smart Campus Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
