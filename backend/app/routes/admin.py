"""
Admin API routes — CRUD for campus_data records + embedding sync.
All endpoints require the X-Admin-Key header matching ADMIN_API_KEY in config.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, Form
from sqlalchemy import text
from app.core.database import get_db_engine
from app.core.config import get_settings
from app.models.schemas import CampusRecordCreate, CampusRecordUpdate
from app.services.document_processor import (
    ensure_documents_table,
    process_pdf,
    embed_and_store_chunks,
)

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


@router.post("/upload")
async def upload_pdf(
    file: UploadFile,
    category: str = Form(...),
    _=Depends(verify_admin),
):
    """
    Upload a PDF, extract its text, chunk it, embed each chunk, and store
    everything in document_chunks so the RAG pipeline can retrieve it.

    Form fields:
      - file:     The PDF file (multipart/form-data).
      - category: One of schedule | general_info | technical_issue |
                  office_hours | exam_schedules | campus_services
    """
    # ── Validate file type ──
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if file.content_type not in ("application/pdf", "application/x-pdf", "binary/octet-stream"):
        # Some browsers send octet-stream, so we accept it if extension is .pdf
        if file.content_type and "pdf" not in file.content_type:
            raise HTTPException(status_code=400, detail="File content type must be PDF.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Ensure documents table exists ──
    ensure_documents_table()

    # ── Extract text and chunk ──
    try:
        result = process_pdf(file_bytes, filename, category)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    chunks = result["chunks"]

    # ── Create document record (get ID first, update count after) ──
    engine = get_db_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO documents (filename, category, chunks_count, uploaded_by) "
                "VALUES (:filename, :category, 0, 'admin') RETURNING id"
            ),
            {"filename": filename, "category": category},
        ).fetchone()
        document_id = row[0]

    # ── Embed and store all chunks ──
    try:
        embed_and_store_chunks(chunks, filename, category, document_id)
    except Exception as e:
        logger.error(f"Embedding failed for doc id={document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # ── Update final chunk count ──
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE documents SET chunks_count = :count WHERE id = :id"),
            {"count": len(chunks), "id": document_id},
        )

    logger.info(
        f"PDF upload complete: '{filename}' → {len(chunks)} chunks, doc id={document_id}"
    )
    return {
        "message": f"Successfully processed '{filename}'",
        "filename": filename,
        "chunks_count": len(chunks),
        "document_id": document_id,
        "category": category,
    }


@router.get("/documents")
async def list_documents(_=Depends(verify_admin)):
    """Return all uploaded PDF documents."""
    ensure_documents_table()
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, filename, category, chunks_count, uploaded_at "
                "FROM documents ORDER BY uploaded_at DESC"
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "category": r[2],
            "chunks_count": r[3],
            "uploaded_at": str(r[4]),
        }
        for r in rows
    ]


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
