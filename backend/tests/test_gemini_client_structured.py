from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError
from pydantic import BaseModel

from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError
from app.services.gemini_client import generate_structured


class DummySchema(BaseModel):
    answer: str
    score: float


def _fake_client(response=None, raises=None) -> MagicMock:
    client = MagicMock()
    if raises is not None:
        client.models.generate_content.side_effect = raises
    else:
        client.models.generate_content.return_value = response
    return client


class TestMissingApiKey:
    def test_raises_when_no_key_configured(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "")

        with pytest.raises(MissingAPIKeyError):
            generate_structured(
                system_instruction="test",
                prompt="test",
                response_model=DummySchema,
            )


class TestSuccessfulStructuredCall:
    def test_returns_parsed_pydantic_instance(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_response = SimpleNamespace(parsed=DummySchema(answer="ok", score=0.9))

        with patch(
            "app.services.gemini_client._get_client",
            return_value=_fake_client(response=fake_response),
        ):
            result = generate_structured(
                system_instruction="test",
                prompt="test",
                response_model=DummySchema,
            )

        assert result == DummySchema(answer="ok", score=0.9)


class TestMalformedOutput:
    def test_raises_when_parsed_is_none(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        fake_response = SimpleNamespace(parsed=None)

        with patch(
            "app.services.gemini_client._get_client",
            return_value=_fake_client(response=fake_response),
        ):
            with pytest.raises(InvalidLLMOutputError):
                generate_structured(
                    system_instruction="test",
                    prompt="test",
                    response_model=DummySchema,
                )


class TestGeminiRequestFailure:
    def test_raises_llm_request_error_on_api_error(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")
        api_error = ClientError(429, {"error": {"message": "rate limited"}}, None)

        with patch(
            "app.services.gemini_client._get_client",
            return_value=_fake_client(raises=api_error),
        ):
            with pytest.raises(LLMRequestError):
                generate_structured(
                    system_instruction="test",
                    prompt="test",
                    response_model=DummySchema,
                )

    def test_raises_llm_request_error_on_network_exception(self, monkeypatch):
        monkeypatch.setattr("app.services.gemini_client.settings.gemini_api_key", "fake-key")

        with patch(
            "app.services.gemini_client._get_client",
            return_value=_fake_client(raises=ConnectionError("network down")),
        ):
            with pytest.raises(LLMRequestError):
                generate_structured(
                    system_instruction="test",
                    prompt="test",
                    response_model=DummySchema,
                )
