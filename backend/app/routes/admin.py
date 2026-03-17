"""
Admin API routes — CRUD for campus_data records + embedding sync.
All endpoints require the X-Admin-Key header matching ADMIN_API_KEY in config.
"""

import csv
import io
import logging
from datetime import date
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, Form, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from app.core.database import get_db_engine
from app.core.config import get_settings
from app.models.schemas import CampusRecordCreate, CampusRecordUpdate, BulkSeedRequest, ChunkUpdate
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
    """Return all uploaded PDF documents with live chunk counts."""
    ensure_documents_table()
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT d.id, d.filename, d.category, "
                "       COUNT(dc.id) AS chunks_count, d.uploaded_at "
                "FROM documents d "
                "LEFT JOIN document_chunks dc "
                "       ON (dc.metadata->>'document_id')::int = d.id "
                "GROUP BY d.id, d.filename, d.category, d.uploaded_at "
                "ORDER BY d.uploaded_at DESC"
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


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, _=Depends(verify_admin)):
    """Delete a document and all its chunks in a single transaction."""
    engine = get_db_engine()
    with engine.begin() as conn:
        chunks_removed = conn.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE (metadata->>'document_id')::int = :id"),
            {"id": document_id},
        ).scalar()

        conn.execute(
            text("DELETE FROM document_chunks WHERE (metadata->>'document_id')::int = :id"),
            {"id": document_id},
        )
        deleted = conn.execute(
            text("DELETE FROM documents WHERE id = :id RETURNING id"),
            {"id": document_id},
        ).fetchone()
        if deleted is None:
            raise HTTPException(status_code=404, detail="Document not found.")

    logger.info(f"Admin deleted document id={document_id}, removed {chunks_removed} chunks")
    return {"message": "Document deleted", "chunks_removed": chunks_removed}


@router.get("/documents/{document_id}/chunks")
async def list_document_chunks(document_id: int, _=Depends(verify_admin)):
    """Return all chunks belonging to a document, ordered by chunk_index."""
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, chunk_index, content, created_at "
                "FROM document_chunks "
                "WHERE (metadata->>'document_id')::int = :doc_id "
                "ORDER BY chunk_index"
            ),
            {"doc_id": document_id},
        ).fetchall()
    return [
        {
            "id": r[0],
            "chunk_index": r[1],
            "content": r[2],
            "created_at": str(r[3]),
        }
        for r in rows
    ]


@router.delete("/chunks/{chunk_id}")
async def delete_chunk(chunk_id: int, _=Depends(verify_admin)):
    """Delete a single document chunk."""
    engine = get_db_engine()
    with engine.begin() as conn:
        deleted = conn.execute(
            text("DELETE FROM document_chunks WHERE id = :id RETURNING id"),
            {"id": chunk_id},
        ).fetchone()
        if deleted is None:
            raise HTTPException(status_code=404, detail="Chunk not found.")
    logger.info(f"Admin deleted chunk id={chunk_id}")
    return {"status": "deleted"}


@router.put("/chunks/{chunk_id}")
async def update_chunk(chunk_id: int, body: ChunkUpdate, _=Depends(verify_admin)):
    """Update a chunk's content and re-generate its embedding."""
    from app.ai.rag import generate_embedding

    new_embedding = generate_embedding(body.content)
    engine = get_db_engine()
    with engine.begin() as conn:
        updated = conn.execute(
            text(
                "UPDATE document_chunks "
                "SET content = :content, embedding = CAST(:embedding AS vector) "
                "WHERE id = :id RETURNING id"
            ),
            {"id": chunk_id, "content": body.content, "embedding": str(new_embedding)},
        ).fetchone()
        if updated is None:
            raise HTTPException(status_code=404, detail="Chunk not found.")
    logger.info(f"Admin updated chunk id={chunk_id}")
    return {"status": "updated"}


@router.get("/interactions")
async def list_interactions(
    category: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _=Depends(verify_admin),
):
    """Return interactions with optional filters, LEFT JOINed with ratings."""
    conditions = []
    params: dict = {"limit": limit, "offset": offset}

    if category:
        conditions.append("i.category = :category")
        params["category"] = category
    if min_confidence is not None:
        conditions.append("i.confidence >= :min_confidence")
        params["min_confidence"] = min_confidence
    if max_confidence is not None:
        conditions.append("i.confidence <= :max_confidence")
        params["max_confidence"] = max_confidence

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT i.id, i.question, i.answer, i.category, i.confidence,
               i.response_time_ms, i.prompt_version, i.created_at,
               r.rating, r.feedback
        FROM interactions i
        LEFT JOIN ratings r ON r.interaction_id = i.id
        {where}
        ORDER BY i.created_at DESC
        LIMIT :limit OFFSET :offset
    """

    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    return [
        {
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "category": r[3],
            "confidence": round(float(r[4]), 3) if r[4] is not None else None,
            "response_time_ms": r[5],
            "prompt_version": r[6],
            "created_at": str(r[7]),
            "rating": r[8],
            "feedback": r[9],
        }
        for r in rows
    ]


@router.get("/stats")
async def get_stats(_=Depends(verify_admin)):
    """Return aggregated interaction statistics."""
    engine = get_db_engine()
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*)                                            AS total_interactions,
                AVG(confidence)                                     AS avg_confidence,
                AVG(response_time_ms)                               AS avg_response_time_ms,
                COUNT(*) FILTER (WHERE confidence < 0.7)           AS low_confidence_count
            FROM interactions
        """)).fetchone()

        cat_rows = conn.execute(text("""
            SELECT category, COUNT(*) AS count
            FROM interactions
            GROUP BY category
            ORDER BY count DESC
        """)).fetchall()

        ratings = conn.execute(text("""
            SELECT
                COUNT(*)                                    AS total_ratings,
                COUNT(*) FILTER (WHERE rating = true)      AS positive_ratings,
                COUNT(*) FILTER (WHERE rating = false)     AS negative_ratings
            FROM ratings
        """)).fetchone()

    return {
        "total_interactions": stats[0] or 0,
        "avg_confidence": round(float(stats[1]), 3) if stats[1] else 0.0,
        "avg_response_time_ms": round(float(stats[2]), 1) if stats[2] else 0.0,
        "low_confidence_count": stats[3] or 0,
        "category_distribution": [
            {"category": r[0], "count": r[1]} for r in cat_rows
        ],
        "total_ratings": ratings[0] or 0,
        "positive_ratings": ratings[1] or 0,
        "negative_ratings": ratings[2] or 0,
    }


@router.get("/export/csv")
async def export_interactions_csv(
    category: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, description="ISO date, e.g. 2025-01-01"),
    date_to: Optional[str] = Query(None, description="ISO date, e.g. 2025-12-31"),
    _=Depends(verify_admin),
):
    """Stream all matching interactions as a CSV file download."""
    conditions = []
    params: dict = {}

    if category:
        conditions.append("i.category = :category")
        params["category"] = category
    if date_from:
        conditions.append("i.created_at >= :date_from")
        params["date_from"] = date_from
    if date_to:
        conditions.append("i.created_at < :date_to")
        params["date_to"] = date_to

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT i.id, i.created_at, i.question, i.answer, i.category,
               i.confidence, i.response_time_ms, i.prompt_version,
               r.rating, r.feedback
        FROM interactions i
        LEFT JOIN ratings r ON r.interaction_id = i.id
        {where}
        ORDER BY i.created_at DESC
    """

    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "question", "answer", "category",
                     "confidence", "response_time_ms", "prompt_version",
                     "rating", "feedback"])
    for r in rows:
        writer.writerow([
            r[0], r[1], r[2], r[3], r[4],
            round(float(r[5]), 3) if r[5] is not None else "",
            r[6] if r[6] is not None else "",
            r[7] or "",
            "positive" if r[8] is True else "negative" if r[8] is False else "",
            r[9] or "",
        ])

    filename = f"interactions_export_{date.today()}.csv"
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/bulk-seed", status_code=201)
async def bulk_seed(body: BulkSeedRequest, _=Depends(verify_admin)):
    """
    Insert up to 100 campus_data records in a single transaction, then
    re-generate embeddings so the RAG pipeline picks them up immediately.
    """
    import json as _json
    from app.services.seed import seed_embeddings

    engine = get_db_engine()
    try:
        with engine.begin() as conn:
            for item in body.records:
                conn.execute(
                    text(
                        "INSERT INTO campus_data (category, title, content, metadata) "
                        "VALUES (:category, :title, :content, CAST(:metadata AS jsonb))"
                    ),
                    {
                        "category": item.category,
                        "title": item.title,
                        "content": item.content,
                        "metadata": _json.dumps(item.metadata),
                    },
                )
    except Exception as e:
        logger.error(f"Bulk seed insert failed, transaction rolled back: {e}")
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")

    try:
        seed_embeddings(engine)
    except Exception as e:
        logger.error(f"Bulk seed embedding sync failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Records inserted but embedding sync failed: {str(e)}",
        )

    n = len(body.records)
    logger.info(f"Bulk seed: inserted {n} records and synced embeddings")
    return {
        "message": f"Successfully imported {n} record{'s' if n != 1 else ''} and synced embeddings.",
        "records_inserted": n,
        "embeddings_synced": True,
    }


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
