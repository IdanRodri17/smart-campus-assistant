"""
Smart Campus Assistant — FastAPI Application
Main entry point for the backend server.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes.ask import router as ask_router
from app.core.config import get_settings

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ──
settings = get_settings()

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


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API info."""
    return {
        "service": "Smart Campus Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
