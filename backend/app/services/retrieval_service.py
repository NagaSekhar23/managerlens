"""Retrieves relevant management-guidance chunks for a manager's situation.

Manager situation -> embed (RETRIEVAL_QUERY) -> vector search over
`knowledge_chunks` -> top-K chunks above a relevance floor.
"""

from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError

from app.db.models import KnowledgeChunk
from app.db.session import SessionLocal
from app.services.errors import KnowledgeRetrievalError
from app.services.gemini_client import embed_text

TOP_K = 4
# Cosine similarity floor below which a chunk is considered "not actually relevant" —
# pgvector's cosine_distance is 1 - cosine_similarity, so similarity = 1 - distance.
MIN_RELEVANCE_SCORE = 0.62


@dataclass(frozen=True)
class KnowledgeMatch:
    title: str
    content: str
    score: float


def _search_chunks(query_embedding: list[float], top_k: int) -> list[tuple[str, str, float]]:
    """Runs the actual pgvector similarity query. Isolated for easy mocking in tests."""
    try:
        with SessionLocal() as session:
            distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
            rows = (
                session.query(
                    KnowledgeChunk.source_title,
                    KnowledgeChunk.content,
                    distance.label("distance"),
                )
                .order_by(distance)
                .limit(top_k)
                .all()
            )
    except SQLAlchemyError as exc:
        raise KnowledgeRetrievalError(f"Knowledge base query failed: {exc}") from exc

    return [(title, content, 1.0 - float(dist)) for title, content, dist in rows]


def retrieve_relevant_knowledge(situation: str, *, top_k: int = TOP_K) -> list[KnowledgeMatch]:
    """Returns the top relevant knowledge-base chunks for `situation`, if any clear the
    relevance floor.

    Raises:
        MissingAPIKeyError / LLMRequestError: embedding the situation failed (propagated
            unchanged — this is a model-provider failure, same as any other Gemini call).
        KnowledgeRetrievalError: the database query itself failed.
    """
    query_embedding = embed_text(situation, task_type="RETRIEVAL_QUERY")

    results = _search_chunks(query_embedding, top_k)

    return [
        KnowledgeMatch(title=title, content=content, score=score)
        for title, content, score in results
        if score >= MIN_RELEVANCE_SCORE
    ]
