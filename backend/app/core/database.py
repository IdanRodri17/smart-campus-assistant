"""
Supabase client initialization and database utilities.
Uses both the Supabase Python client (for auth/storage) and
direct PostgreSQL via SQLAlchemy (for pgvector queries).
"""

from supabase import create_client, Client
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from functools import lru_cache


@lru_cache()
def get_supabase_client() -> Client:
    """Get the Supabase client for API operations."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@lru_cache()
def get_db_engine():
    """Get SQLAlchemy engine for direct PostgreSQL queries (pgvector)."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_db_session():
    """Create a new database session."""
    engine = get_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_db_health() -> bool:
    """Health check — verify database connection."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
