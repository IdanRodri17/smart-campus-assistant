# ══════════════════════════════════════════
# Smart Campus Assistant — Multi-stage Dockerfile
# Stage 1: Install dependencies
# Stage 2: Production image (smaller, non-root)
# ══════════════════════════════════════════

# ── Stage 1: Builder ──
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# ── Stage 2: Production ──
FROM python:3.12-slim AS production

# Security: create non-root user
RUN groupadd -r campus && useradd -r -g campus -s /bin/false campususer

# Install only runtime dependencies (libpq for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/app ./app

# Switch to non-root user
USER campususer

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
