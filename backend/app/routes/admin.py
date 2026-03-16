"""
Admin API routes — CRUD for campus_data records + embedding sync.
All endpoints require the X-Admin-Key header matching ADMIN_API_KEY in config.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Header
from sqlalchemy import text
from app.core.database import get_db_engine
from app.core.config import get_settings
from app.models.schemas import CampusRecordCreate, CampusRecordUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


async def verify_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    """FastAPI dependency — rejects requests with wrong or missing admin key."""
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Admin access is not configured on this server.",
        )
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key.")


@router.get("/records")
async def list_records(_=Depends(verify_admin)):
    """Return all campus_data records ordered by category and title."""
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, category, title, content, updated_at "
                "FROM campus_data ORDER BY category, title"
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "category": r[1],
            "title": r[2],
            "content": r[3],
            "updated_at": str(r[4]),
        }
        for r in rows
    ]


@router.post("/records", status_code=201)
async def create_record(body: CampusRecordCreate, _=Depends(verify_admin)):
    """Create a new campus data record."""
    engine = get_db_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO campus_data (category, title, content, metadata) "
                "VALUES (:category, :title, :content, '{}') RETURNING id"
            ),
            {"category": body.category, "title": body.title, "content": body.content},
        )
        new_id = result.fetchone()[0]
    logger.info(f"Admin created campus record id={new_id}: '{body.title}'")
    return {"id": new_id, "status": "created"}


@router.put("/records/{record_id}")
async def update_record(
    record_id: int, body: CampusRecordUpdate, _=Depends(verify_admin)
):
    """Update an existing campus data record. Only supplied fields are changed."""
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE campus_data "
                "SET category   = COALESCE(:category, category), "
                "    title      = COALESCE(:title,    title), "
                "    content    = COALESCE(:content,  content), "
                "    updated_at = NOW() "
                "WHERE id = :id"
            ),
            {
                "id": record_id,
                "category": body.category,
                "title": body.title,
                "content": body.content,
            },
        )
    logger.info(f"Admin updated campus record id={record_id}")
    return {"status": "updated"}


@router.delete("/records/{record_id}")
async def delete_record(record_id: int, _=Depends(verify_admin)):
    """Delete a campus data record."""
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM campus_data WHERE id = :id"), {"id": record_id})
    logger.info(f"Admin deleted campus record id={record_id}")
    return {"status": "deleted"}


@router.post("/sync-embeddings")
async def sync_embeddings(_=Depends(verify_admin)):
    """
    Re-generate embeddings for all campus_data records.
    Call this after adding or editing records so RAG results stay up to date.
    """
    from app.services.seed import seed_embeddings

    engine = get_db_engine()
    try:
        seed_embeddings(engine)
        logger.info("Admin triggered embedding sync — complete")
        return {"status": "ok", "message": "Embeddings synced successfully."}
    except Exception as e:
        logger.error(f"Embedding sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
