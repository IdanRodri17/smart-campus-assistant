"""
PDF document processor — UC4 from SRS.

Handles:
  1. Text extraction from PDF bytes using pdfplumber
  2. Token-aware chunking with overlap using tiktoken
  3. Embedding each chunk via the existing OpenAI singleton
  4. Storing chunks in document_chunks table (same schema used by seed.py)
  5. Managing the documents table (one row per uploaded PDF)
"""

import io
import json
import logging

import pdfplumber
import tiktoken
from sqlalchemy import text

from app.ai.rag import generate_embedding
from app.core.database import get_db_engine

logger = logging.getLogger(__name__)

# cl100k_base is the tokeniser used by text-embedding-3-small
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500   # target chunk size
OVERLAP_TOKENS = 50  # tokens shared between consecutive chunks


# ── Table setup ──────────────────────────────────────────────────────────────

def ensure_documents_table() -> None:
    """Create the documents table if it does not already exist."""
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id           SERIAL PRIMARY KEY,
                filename     VARCHAR(255) NOT NULL,
                category     VARCHAR(50),
                chunks_count INTEGER DEFAULT 0,
                uploaded_at  TIMESTAMP DEFAULT NOW(),
                uploaded_by  VARCHAR(50) DEFAULT 'admin'
            )
            """
        ))


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text_input: str) -> list[str]:
    """
    Split text into token-bounded chunks with overlap.

    Strategy:
      - Tokenise the full text with tiktoken (matches the embedding model).
      - Slide a window of MAX_TOKENS tokens, stepping by (MAX_TOKENS - OVERLAP_TOKENS).
      - Decode each window back to a string.

    This guarantees no chunk ever exceeds the embedding model's token limit and
    that consecutive chunks share OVERLAP_TOKENS tokens of context so the RAG
    retriever can find answers that span chunk boundaries.
    """
    tokens = _TOKENIZER.encode(text_input)
    chunks: list[str] = []
    step = MAX_TOKENS - OVERLAP_TOKENS
    start = 0

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        chunk_text = _TOKENIZER.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += step

    return chunks


# ── PDF extraction ────────────────────────────────────────────────────────────

def process_pdf(file_bytes: bytes, filename: str, category: str) -> dict:
    """
    Extract text from a PDF and split it into embeddable chunks.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename:   Original filename (used for logging and metadata).
        category:   Admin-selected category tag for RAG filtering.

    Returns:
        {"filename": str, "chunks": list[str], "category": str}

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    pages: list[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append(page_text.strip())
            else:
                logger.debug(f"Page {page_num} of '{filename}' had no extractable text")

    if not pages:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "The PDF may be scanned/image-only."
        )

    full_text = "\n\n".join(pages)
    chunks = _chunk_text(full_text)

    logger.info(
        f"Processed '{filename}': {len(pages)} pages → {len(chunks)} chunks "
        f"(category={category})"
    )
    return {"filename": filename, "chunks": chunks, "category": category}


# ── Embedding + storage ───────────────────────────────────────────────────────

def embed_and_store_chunks(
    chunks: list[str],
    filename: str,
    category: str,
    document_id: int,
) -> None:
    """
    Embed each chunk and insert it into document_chunks.

    Uses the same `generate_embedding()` from rag.py — this means:
      - The shared OpenAI singleton client is reused.
      - Redis cache is consulted before calling the API (cache hit = free).

    All inserts are wrapped in a single engine.begin() transaction so the
    database is never left in a partial state if an embedding call fails.

    Args:
        chunks:      List of text chunks from process_pdf().
        filename:    Original PDF filename (stored in source column).
        category:    Category tag stored in metadata JSONB.
        document_id: FK to documents.id — stored in metadata JSONB.
    """
    engine = get_db_engine()

    with engine.begin() as conn:
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            conn.execute(
                text(
                    """
                    INSERT INTO document_chunks
                        (content, embedding, metadata, source, chunk_index)
                    VALUES
                        (:content, CAST(:embedding AS vector), :metadata, :source, :chunk_index)
                    """
                ),
                {
                    "content": chunk,
                    "embedding": str(embedding),
                    "metadata": json.dumps({
                        "category": category,
                        "document_id": document_id,
                        "source": "pdf_upload",
                        "filename": filename,
                    }),
                    "source": filename,
                    "chunk_index": i,
                },
            )
            logger.debug(f"Stored chunk {i + 1}/{len(chunks)} for '{filename}'")

    logger.info(f"embed_and_store_chunks complete: {len(chunks)} chunks for doc id={document_id}")
