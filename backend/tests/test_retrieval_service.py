from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.db.session import engine
from app.services.errors import KnowledgeRetrievalError
from app.services.retrieval_service import (
    MIN_RELEVANCE_SCORE,
    KnowledgeMatch,
    retrieve_relevant_knowledge,
)


def _knowledge_base_is_populated() -> bool:
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM knowledge_chunks")).scalar()
        return bool(count and count > 0)
    except SQLAlchemyError:
        return False


requires_live_knowledge_base = pytest.mark.skipif(
    not settings.gemini_api_key or not _knowledge_base_is_populated(),
    reason="Requires GEMINI_API_KEY and an ingested knowledge_chunks table",
)


def _patch_embed(monkeypatch, vector=None):
    monkeypatch.setattr(
        "app.services.retrieval_service.embed_text",
        lambda situation, task_type: vector or [0.1, 0.2, 0.3],
    )


class TestRetrievalRanking:
    def test_returns_matches_ordered_by_relevance(self, monkeypatch):
        _patch_embed(monkeypatch)

        with patch(
            "app.services.retrieval_service._search_chunks",
            return_value=[
                ("Low Match", "some text", MIN_RELEVANCE_SCORE + 0.05),
                ("Best Match", "some text", 0.90),
                ("Mid Match", "some text", 0.75),
            ],
        ):
            matches = retrieve_relevant_knowledge("a situation")

        assert [m.title for m in matches] == ["Low Match", "Best Match", "Mid Match"]

    def test_respects_top_k_from_search(self, monkeypatch):
        _patch_embed(monkeypatch)
        with patch(
            "app.services.retrieval_service._search_chunks",
            return_value=[("Only Match", "text", 0.95)],
        ) as mock_search:
            retrieve_relevant_knowledge("a situation", top_k=2)

        args, _ = mock_search.call_args
        assert args[1] == 2


class TestNoRelevantResults:
    def test_filters_out_matches_below_relevance_floor(self, monkeypatch):
        _patch_embed(monkeypatch)
        with patch(
            "app.services.retrieval_service._search_chunks",
            return_value=[
                ("Weak Match", "text", MIN_RELEVANCE_SCORE - 0.05),
                ("Also Weak", "text", 0.1),
            ],
        ):
            matches = retrieve_relevant_knowledge("an unrelated situation")

        assert matches == []

    def test_empty_search_results_returns_empty_list(self, monkeypatch):
        _patch_embed(monkeypatch)
        with patch("app.services.retrieval_service._search_chunks", return_value=[]):
            matches = retrieve_relevant_knowledge("a situation")

        assert matches == []


class TestLiveRetrieval:
    @requires_live_knowledge_base
    def test_retrieves_topically_relevant_chunks_from_real_knowledge_base(self):
        situation = (
            "My engineer has missed three deadlines and has become quiet during team "
            "meetings. I am worried they are disengaged."
        )

        matches = retrieve_relevant_knowledge(situation)

        assert len(matches) > 0
        assert all(m.score >= MIN_RELEVANCE_SCORE for m in matches)
        # Results must actually be sorted by relevance, most relevant first.
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)
        # At least one of the topically-relevant docs should surface for this situation.
        titles = {m.title for m in matches}
        assert titles & {"Performance Conversations", "One-on-Ones", "Difficult Conversations"}


class TestDatabaseFailure:
    def test_query_failure_raises_knowledge_retrieval_error(self, monkeypatch):
        _patch_embed(monkeypatch)

        def _raise(*args, **kwargs):
            raise SQLAlchemyError("connection refused")

        with patch("app.services.retrieval_service.SessionLocal", side_effect=_raise):
            with pytest.raises(KnowledgeRetrievalError):
                retrieve_relevant_knowledge("a situation")
