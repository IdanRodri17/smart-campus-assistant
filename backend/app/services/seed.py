"""
Database seed script — creates tables, enables pgvector, and populates campus data.
Run this once to set up the database with initial campus information.

Usage: python -m app.services.seed
"""

import json
import logging
from sqlalchemy import text
from app.core.database import get_db_engine
from app.ai.rag import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# SQL Schema
# ══════════════════════════════════════════

SCHEMA_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Campus structured data
CREATE TABLE IF NOT EXISTS campus_data (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Document chunks with vector embeddings (for RAG)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    source VARCHAR(200),
    chunk_index INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Interaction logs (questions + answers)
CREATE TABLE IF NOT EXISTS interactions (
    id VARCHAR(36) PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    category VARCHAR(50),
    confidence FLOAT,
    response_time_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,
    prompt_version VARCHAR(10),
    is_fallback BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User ratings
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    interaction_id VARCHAR(36) REFERENCES interactions(id),
    rating BOOLEAN NOT NULL,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- Create index for category lookups
CREATE INDEX IF NOT EXISTS idx_campus_data_category
    ON campus_data(category);
"""


# ══════════════════════════════════════════
# Seed Data — Campus Information
# ══════════════════════════════════════════

CAMPUS_DATA = [
    # ── Schedule ──
    {
        "category": "schedule",
        "title": "Dr. Sarah Cohen - Office Hours",
        "content": "Dr. Sarah Cohen holds office hours every Sunday and Tuesday from 10:00 to 12:00 in Building B, Room 305. Appointments can be made via email: s.cohen@campus.ac.il",
    },
    {
        "category": "schedule",
        "title": "Dr. Yossi Levi - Office Hours",
        "content": "Dr. Yossi Levi is available for student meetings on Monday and Wednesday from 14:00 to 16:00 in Building A, Room 210. Walk-ins welcome, no appointment needed.",
    },
    {
        "category": "schedule",
        "title": "Python Foundations Final Exam",
        "content": "The Python Foundations final exam is scheduled for March 20, 2026 at 09:00 in Hall C, Room 101. Duration: 3 hours. Bring your student ID. The exam is open-book (digital notes allowed, no internet).",
    },
    {
        "category": "schedule",
        "title": "SQL Module Exam",
        "content": "The Databases & SQL exam takes place on March 25, 2026 at 10:00 in Computer Lab 3, Building D. Duration: 2 hours. Practical exam using PostgreSQL — make sure your laptop has pgAdmin installed.",
    },
    {
        "category": "schedule",
        "title": "AI Overview Final Project Submission",
        "content": "AI Overview & Tools final project is due April 1, 2026 by 23:59. Submit via the course portal. Late submissions: -10 points per day, maximum 3 days late.",
    },
    {
        "category": "schedule",
        "title": "Weekly Class Schedule - CyberPro Bootcamp",
        "content": "Classes are held Sunday through Thursday, 09:00-17:00 with a lunch break from 12:00-13:00. Friday classes are optional review sessions from 09:00-12:00.",
    },
    # ── General Info ──
    {
        "category": "general_info",
        "title": "Campus Library Hours",
        "content": "The campus library is open Sunday-Thursday 08:00-22:00, Friday 08:00-14:00. The library has 3 quiet study rooms that can be reserved via the student portal for up to 3 hours per booking.",
    },
    {
        "category": "general_info",
        "title": "Student Parking",
        "content": "Student parking is available in Parking Lot B (behind Building D). A monthly parking permit costs 150 NIS and can be purchased at the admin office. Lot opens at 07:00 and closes at 23:00.",
    },
    {
        "category": "general_info",
        "title": "Cafeteria Information",
        "content": "The main cafeteria is located on the ground floor of Building A. Operating hours: Sunday-Thursday 07:30-19:00. The cafeteria offers kosher meals, vegan options, and a coffee bar. Student meal plan: 35 NIS per day.",
    },
    {
        "category": "general_info",
        "title": "Campus WiFi",
        "content": "Connect to 'CyberPro-Student' WiFi network. Login with your student email and password. Coverage: all buildings and outdoor common areas. Speed: up to 100 Mbps download. VPN access available for remote lab work.",
    },
    {
        "category": "general_info",
        "title": "Campus Locations Guide",
        "content": "Building A: Administration, classrooms, cafeteria. Building B: Faculty offices, meeting rooms. Building C: Lecture halls and auditorium. Building D: Computer labs, server room. Building E: Student lounge and study areas.",
    },
    {
        "category": "general_info",
        "title": "Course Registration Process",
        "content": "Course registration opens 2 weeks before each module starts. Register through the student portal under 'My Courses'. Each student must complete prerequisite modules before advancing. Contact academic advisor for special approvals.",
    },
    # ── Technical Issues ──
    {
        "category": "technical_issue",
        "title": "Student Portal Login Issues",
        "content": "If you cannot log into the student portal, try resetting your password at portal.campus.ac.il/reset. If the issue persists, contact IT support at it.support@campus.ac.il or call extension 1234. IT office hours: Sun-Thu 08:30-17:00.",
    },
    {
        "category": "technical_issue",
        "title": "Computer Lab Access",
        "content": "Computer labs are accessible with your student card. Lab 1-2 (Building D, 2nd floor): open access Sun-Thu 08:00-21:00. Lab 3: reserved for exams and supervised sessions only. Software issues: report to lab technician or email lab.support@campus.ac.il.",
    },
    {
        "category": "technical_issue",
        "title": "Printing Services",
        "content": "Printers are available in the library and Computer Lab 1. Each student gets 100 free pages per month. Additional pages: 0.30 NIS per page. For printing issues, check paper tray first, then contact library staff. Color printing available in Library only.",
    },
    {
        "category": "technical_issue",
        "title": "VPN Setup for Remote Access",
        "content": "To access campus resources from home, install the FortiClient VPN. Download from portal.campus.ac.il/vpn. Server: vpn.campus.ac.il. Use your student credentials. VPN gives access to: internal databases, lab machines (via SSH), and course materials server.",
    },
]


def create_schema(engine):
    """Create database tables and enable pgvector."""
    logger.info("Creating database schema...")
    with engine.connect() as conn:
        for statement in SCHEMA_SQL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    logger.info("Schema created successfully")


def seed_campus_data(engine):
    """Insert structured campus data."""
    logger.info("Seeding campus data...")
    with engine.connect() as conn:
        # Clear existing data (idempotent)
        conn.execute(text("DELETE FROM campus_data"))

        for item in CAMPUS_DATA:
            conn.execute(
                text(
                    """
                    INSERT INTO campus_data (category, title, content, metadata)
                    VALUES (:category, :title, :content, :metadata)
                """
                ),
                {
                    "category": item["category"],
                    "title": item["title"],
                    "content": item["content"],
                    "metadata": json.dumps({"source": "seed_data"}),
                },
            )
        conn.commit()
    logger.info(f"Seeded {len(CAMPUS_DATA)} campus data records")


def seed_embeddings(engine):
    """Generate embeddings for campus data and store in document_chunks."""
    logger.info("Generating embeddings for RAG... (this may take a minute)")
    with engine.connect() as conn:
        # Clear existing chunks (idempotent)
        conn.execute(text("DELETE FROM document_chunks"))

        for i, item in enumerate(CAMPUS_DATA):
            # Combine title and content for embedding
            text_to_embed = f"{item['title']}: {item['content']}"

            try:
                embedding = generate_embedding(text_to_embed)

                conn.execute(
                    text(
                        """
                        INSERT INTO document_chunks (content, embedding, metadata, source, chunk_index)
                        VALUES (:content, :embedding::vector, :metadata, :source, :chunk_index)
                    """
                    ),
                    {
                        "content": text_to_embed,
                        "embedding": str(embedding),
                        "metadata": json.dumps(
                            {
                                "category": item["category"],
                                "title": item["title"],
                            }
                        ),
                        "source": "campus_seed_data",
                        "chunk_index": i,
                    },
                )
                logger.info(f"  [{i+1}/{len(CAMPUS_DATA)}] Embedded: {item['title']}")

            except Exception as e:
                logger.error(
                    f"  [{i+1}/{len(CAMPUS_DATA)}] Failed to embed '{item['title']}': {e}"
                )

        conn.commit()
    logger.info("Embedding generation complete")


def run_seed():
    """Run the full seed process."""
    engine = get_db_engine()

    create_schema(engine)
    seed_campus_data(engine)
    seed_embeddings(engine)

    logger.info("Seed complete! Database is ready.")


if __name__ == "__main__":
    run_seed()
