"""
RAG (Retrieval-Augmented Generation) service.
Handles embedding generation and vector similarity search via Supabase pgvector.

Supports metadata filtering by category (FR-03) for improved retrieval precision,
with automatic fallback to unfiltered search if too few results are found (SRS 9.5).
"""

import logging
import json
from openai import OpenAI
from sqlalchemy import text
from app.core.config import get_settings
from app.core.database import get_db_engine
from app.core.cache import cache_get, cache_set, EMBEDDING_TTL

logger = logging.getLogger(__name__)

# Module-level singleton — created once, reused for every request
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key)
    return _client


def generate_embedding(text_input: str) -> list[float]:
    """
    Generate a vector embedding for a text input using OpenAI.
    Checks Redis cache first — a cache hit avoids the API call entirely.

    Args:
        text_input: Text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    cached = cache_get("emb", text_input)
    if cached is not None:
        logger.debug("Embedding cache hit")
        return cached

    settings = get_settings()
    response = _get_client().embeddings.create(
        model=settings.embedding_model,
        input=text_input,
    )
    embedding = response.data[0].embedding

    cache_set("emb", text_input, embedding, EMBEDDING_TTL)
    return embedding


def search_similar_chunks(
    query_embedding: list[float],
    category: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search for the most similar document chunks using pgvector cosine similarity.

    Implements two-pass retrieval strategy (SRS Section 9.5):
    - Pass 1: filter by category metadata for higher precision
    - Pass 2: if fewer than top_k results, fallback to unfiltered search for recall
    """
    engine = get_db_engine()

    # ── Pass 1: Filtered search (if category provided) ──
    if category:
        filtered_query = text(
            """
            SELECT 
                content,
                1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity,
                metadata
            FROM document_chunks
            WHERE metadata->>'category' = :category
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT :top_k
            """
        )

        try:
            with engine.connect() as conn:
                results = conn.execute(
                    filtered_query,
                    {
                        "query_vector": str(query_embedding),
                        "category": category,
                        "top_k": top_k,
                    },
                ).fetchall()

            if len(results) >= top_k:
                chunks = _rows_to_chunks(results)
                logger.info(
                    f"RAG filtered search ({category}): {len(chunks)} chunks "
                    f"(top similarity: {chunks[0]['similarity']:.3f})"
                )
                return chunks
            else:
                logger.info(
                    f"RAG filtered search ({category}): only {len(results)} results, "
                    f"falling back to unfiltered search"
                )
        except Exception as e:
            logger.warning(f"Filtered vector search failed: {e}, trying unfiltered")

    # ── Pass 2: Unfiltered search (fallback or no category) ──
    unfiltered_query = text(
        """
        SELECT 
            content,
            1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity,
            metadata
        FROM document_chunks
        ORDER BY embedding <=> CAST(:query_vector AS vector)
        LIMIT :top_k
        """
    )

    try:
        with engine.connect() as conn:
            results = conn.execute(
                unfiltered_query,
                {
                    "query_vector": str(query_embedding),
                    "top_k": top_k,
                },
            ).fetchall()

        chunks = _rows_to_chunks(results)
        logger.info(
            f"RAG unfiltered search: {len(chunks)} chunks"
            + (f" (top similarity: {chunks[0]['similarity']:.3f})" if chunks else "")
        )
        return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def _rows_to_chunks(results) -> list[dict]:
    """Convert database rows to chunk dicts."""
    chunks = []
    for row in results:
        metadata = row[2]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        chunks.append(
            {
                "content": row[0],
                "similarity": float(row[1]),
                "metadata": metadata if metadata else {},
            }
        )
    return chunks


def retrieve_context(
    question: str,
    category: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Full RAG retrieval pipeline: embed question → search pgvector → return context.
    """
    settings = get_settings()
    k = top_k or settings.rag_top_k

    try:
        embedding = generate_embedding(question)
        chunks = search_similar_chunks(embedding, category=category, top_k=k)
        return chunks

    except Exception as e:
        logger.error(f"RAG retrieval pipeline failed: {e}")
        return []


def get_structured_data(category: str) -> list[dict]:
    """
    Fetch structured campus data filtered by category.
    This complements the RAG vector search with exact-match structured data.

    Args:
        category: The question category (schedule, general_info, technical_issue).

    Returns:
        List of structured data records.
    """
    engine = get_db_engine()

    query = text(
        """
        SELECT title, content, metadata
        FROM campus_data
        WHERE category = :category
        ORDER BY updated_at DESC
        LIMIT 10
    """
    )

    try:
        with engine.connect() as conn:
            results = conn.execute(query, {"category": category}).fetchall()

        return [
            {"title": row[0], "content": row[1], "metadata": row[2]} for row in results
        ]

    except Exception as e:
        logger.error(f"Structured data fetch failed: {e}")
        return []
