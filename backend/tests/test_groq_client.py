"""Tests for the Groq reasoning/tool-calling client (`app/services/groq_client.py`).

Groq is always mocked here via `_get_client` — these tests never make a live call.
Covers: client configuration (A), structured generation (B), tool calling (C), and the
max_tool_calls ceiling (D).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError
from app.services.gemini_client import ToolDeclaration
from app.services.groq_client import generate_structured, run_tool_loop


class DummySchema(BaseModel):
    answer: str
    score: float


def _completion(content: str | None = None, tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _client_with_responses(*responses) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = list(responses)
    return client


def _echo_tool(name: str, result="ok") -> ToolDeclaration:
    return ToolDeclaration(
        name=name,
        description=f"test tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        handler=lambda _args: result,
    )


# --- A. Groq client configuration -------------------------------------------------


class TestClientConfiguration:
    def test_missing_api_key_raises_before_any_request(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "")

        with pytest.raises(MissingAPIKeyError):
            generate_structured(
                system_instruction="test", prompt="test", response_model=DummySchema
            )

    def test_client_is_constructed_with_configured_api_key_and_model(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        monkeypatch.setattr("app.services.groq_client.settings.groq_model", "openai/gpt-oss-120b")
        monkeypatch.setattr("app.services.groq_client._client", None)

        captured = {}

        class FakeGroq:
            def __init__(self, api_key, timeout):
                captured["api_key"] = api_key
                captured["timeout"] = timeout
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: _completion(
                            content=json.dumps({"answer": "ok", "score": 0.5})
                        )
                    )
                )

        with patch("app.services.groq_client.Groq", FakeGroq):
            result = generate_structured(
                system_instruction="test", prompt="test", response_model=DummySchema
            )

        assert captured["api_key"] == "fake-key"
        assert result == DummySchema(answer="ok", score=0.5)


# --- B. Structured generation -------------------------------------------------------


class TestGenerateStructured:
    def test_returns_parsed_pydantic_instance(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        response = _completion(content=json.dumps({"answer": "ok", "score": 0.9}))

        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(response),
        ):
            result = generate_structured(
                system_instruction="test", prompt="test", response_model=DummySchema
            )

        assert result == DummySchema(answer="ok", score=0.9)

    def test_request_uses_json_schema_response_format_and_no_tools(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        response = _completion(content=json.dumps({"answer": "ok", "score": 0.9}))
        client = _client_with_responses(response)

        with patch("app.services.groq_client._get_client", return_value=client):
            generate_structured(
                system_instruction="sys", prompt="usr", response_model=DummySchema
            )

        _, kwargs = client.chat.completions.create.call_args
        assert "tools" not in kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "DummySchema"
        assert kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_raises_when_content_is_empty(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")

        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(_completion(content=None)),
        ):
            with pytest.raises(InvalidLLMOutputError):
                generate_structured(
                    system_instruction="test", prompt="test", response_model=DummySchema
                )

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")

        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(_completion(content="not json")),
        ):
            with pytest.raises(InvalidLLMOutputError):
                generate_structured(
                    system_instruction="test", prompt="test", response_model=DummySchema
                )

    def test_raises_on_schema_validation_failure(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        response = _completion(content=json.dumps({"answer": "ok"}))  # missing "score"

        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(response),
        ):
            with pytest.raises(InvalidLLMOutputError):
                generate_structured(
                    system_instruction="test", prompt="test", response_model=DummySchema
                )

    def test_raises_llm_request_error_on_network_exception(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("network down")

        with patch("app.services.groq_client._get_client", return_value=client):
            with pytest.raises(LLMRequestError):
                generate_structured(
                    system_instruction="test", prompt="test", response_model=DummySchema
                )


# --- C. Tool calling -----------------------------------------------------------------


class TestRunToolLoop:
    def test_no_tools_declared_returns_empty_without_calling_groq(self):
        with patch("app.services.groq_client._get_client") as mock_get_client:
            records = run_tool_loop(
                system_instruction="test", prompt="test", tools=[], max_tool_calls=5
            )

        assert records == []
        mock_get_client.assert_not_called()

    def test_no_tool_call_requested_returns_empty(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(_completion(content="no tools needed")),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee")],
                max_tool_calls=5,
            )

        assert records == []

    def test_executes_allowlisted_tool_and_returns_record(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "get_employee", {})]),
            _completion(content="done"),
        ]
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee", result={"name": "Jordan Alvarez"})],
                max_tool_calls=5,
            )

        assert len(records) == 1
        assert records[0].name == "get_employee"
        assert records[0].result == {"name": "Jordan Alvarez"}
        assert records[0].error is None

    def test_executes_multiple_tools_across_turns(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "get_employee", {})]),
            _completion(tool_calls=[_tool_call("call_2", "get_jira_activity", {})]),
            _completion(content="done"),
        ]
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee"), _echo_tool("get_jira_activity")],
                max_tool_calls=5,
            )

        assert [r.name for r in records] == ["get_employee", "get_jira_activity"]

    def test_executes_multiple_tools_requested_in_one_turn(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        responses = [
            _completion(
                tool_calls=[
                    _tool_call("call_1", "get_employee", {}),
                    _tool_call("call_2", "get_jira_activity", {}),
                ]
            ),
            _completion(content="done"),
        ]
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee"), _echo_tool("get_jira_activity")],
                max_tool_calls=5,
            )

        assert {r.name for r in records} == {"get_employee", "get_jira_activity"}

    def test_unknown_tool_is_rejected_not_executed(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        handler = MagicMock()
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "delete_database", {})]),
            _completion(content="done"),
        ]
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[
                    ToolDeclaration(
                        name="get_employee",
                        description="allowed",
                        parameters_schema={"type": "object", "properties": {}},
                        handler=handler,
                    )
                ],
                max_tool_calls=5,
            )

        handler.assert_not_called()
        assert len(records) == 1
        assert records[0].name == "delete_database"
        assert records[0].result is None
        assert "not" in records[0].error.lower()

    def test_handler_exception_is_captured_not_raised(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")

        def _boom(_args):
            raise ValueError("bad input")

        responses = [
            _completion(tool_calls=[_tool_call("call_1", "get_employee", {})]),
            _completion(content="done"),
        ]
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[
                    ToolDeclaration(
                        name="get_employee",
                        description="test",
                        parameters_schema={"type": "object", "properties": {}},
                        handler=_boom,
                    )
                ],
                max_tool_calls=5,
            )

        assert len(records) == 1
        assert records[0].error == "bad input"

    def test_request_passes_openai_style_function_tools(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        client = _client_with_responses(_completion(content="done"))

        with patch("app.services.groq_client._get_client", return_value=client):
            run_tool_loop(
                system_instruction="sys",
                prompt="usr",
                tools=[_echo_tool("get_employee")],
                max_tool_calls=5,
            )

        _, kwargs = client.chat.completions.create.call_args
        assert "response_format" not in kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_employee",
                    "description": "test tool get_employee",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


# --- D. max_tool_calls enforcement --------------------------------------------------


class TestMaxToolCallLimit:
    def test_stops_executing_once_limit_reached(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        responses = [
            _completion(tool_calls=[_tool_call(f"call_{i}", "get_employee", {})])
            for i in range(10)
        ]
        client = _client_with_responses(*responses)

        with patch("app.services.groq_client._get_client", return_value=client):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee")],
                max_tool_calls=2,
            )

        executed = [r for r in records if r.error is None]
        assert len(executed) == 2
        assert client.chat.completions.create.call_count <= 3

    def test_limit_of_zero_never_executes_anything(self, monkeypatch):
        monkeypatch.setattr("app.services.groq_client.settings.groq_api_key", "fake-key")
        handler = MagicMock()
        with patch(
            "app.services.groq_client._get_client",
            return_value=_client_with_responses(
                _completion(tool_calls=[_tool_call("call_1", "get_employee", {})])
            ),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[
                    ToolDeclaration(
                        name="get_employee",
                        description="test",
                        parameters_schema={"type": "object", "properties": {}},
                        handler=handler,
                    )
                ],
                max_tool_calls=0,
            )

        handler.assert_not_called()
        assert records == []
