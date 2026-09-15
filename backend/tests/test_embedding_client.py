from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError

from app.services.errors import LLMRequestError, MissingAPIKeyError
from app.services.gemini_client import embed_text


def _fake_embedding_response(values: list[float]):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


class TestMissingApiKey:
    def test_raises_when_no_key_configured(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "")

        with pytest.raises(MissingAPIKeyError):
            embed_text("some text", task_type="RETRIEVAL_QUERY")


class TestSuccessfulEmbedding:
    def test_returns_list_of_floats(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = _fake_embedding_response([0.1, 0.2, 0.3])

        with patch("app.services.gemini_client._get_client", return_value=fake_client):
            result = embed_text("some text", task_type="RETRIEVAL_DOCUMENT")

        assert result == [0.1, 0.2, 0.3]
        _, kwargs = fake_client.models.embed_content.call_args
        assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

    def test_uses_different_task_type_for_queries(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = _fake_embedding_response([0.4, 0.5])

        with patch("app.services.gemini_client._get_client", return_value=fake_client):
            embed_text("a situation", task_type="RETRIEVAL_QUERY")

        _, kwargs = fake_client.models.embed_content.call_args
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"


class TestEmbeddingFailures:
    def test_raises_llm_request_error_on_api_error(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_client = MagicMock()
        fake_client.models.embed_content.side_effect = ClientError(
            429, {"error": {"message": "rate limited"}}, None
        )

        with patch("app.services.gemini_client._get_client", return_value=fake_client):
            with pytest.raises(LLMRequestError):
                embed_text("some text", task_type="RETRIEVAL_QUERY")

    def test_raises_llm_request_error_on_empty_embedding(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = SimpleNamespace(embeddings=[])

        with patch("app.services.gemini_client._get_client", return_value=fake_client):
            with pytest.raises(LLMRequestError):
                embed_text("some text", task_type="RETRIEVAL_QUERY")
