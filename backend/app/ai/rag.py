"""
RAG (Retrieval-Augmented Generation) service.
Handles embedding generation and vector similarity search via Supabase pgvector.
"""

import logging
from openai import OpenAI
from sqlalchemy import text
from app.core.config import get_settings
from app.core.database import get_db_engine

logger = logging.getLogger(__name__)


def generate_embedding(text_input: str) -> list[float]:
    """
    Generate a vector embedding for a text input using OpenAI.

    Args:
        text_input: Text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text_input,
    )

    return response.data[0].embedding


def search_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Search for the most similar document chunks using pgvector cosine similarity.

    Args:
        query_embedding: The query's embedding vector.
        top_k: Number of results to return.

    Returns:
        List of dicts with 'content', 'similarity', and 'metadata'.
    """
    settings = get_settings()
    engine = get_db_engine()

    # pgvector cosine distance: <=> operator (lower = more similar)
    # We convert to similarity: 1 - distance
    query = text(
        """
        SELECT 
            content,
            1 - (embedding <=> :query_vector::vector) AS similarity,
            metadata
        FROM document_chunks
        ORDER BY embedding <=> :query_vector::vector
        LIMIT :top_k
    """
    )

    try:
        with engine.connect() as conn:
            results = conn.execute(
                query,
                {
                    "query_vector": str(query_embedding),
                    "top_k": top_k,
                },
            ).fetchall()

        chunks = []
        for row in results:
            chunks.append(
                {
                    "content": row[0],
                    "similarity": float(row[1]),
                    "metadata": row[2] if row[2] else {},
                }
            )

        logger.info(
            f"RAG search returned {len(chunks)} chunks (top similarity: {chunks[0]['similarity']:.3f})"
            if chunks
            else "RAG search returned 0 chunks"
        )
        return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def retrieve_context(question: str, top_k: int | None = None) -> list[dict]:
    """
    Full RAG retrieval pipeline: embed question → search pgvector → return context.

    Args:
        question: The student's question.
        top_k: Override for number of results (uses config default if None).

    Returns:
        List of relevant context chunks.
    """
    settings = get_settings()
    k = top_k or settings.rag_top_k

    try:
        # Step 1: Embed the question
        embedding = generate_embedding(question)

        # Step 2: Search for similar chunks
        chunks = search_similar_chunks(embedding, top_k=k)

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
