"""Tests for the generic Gemini tool-calling loop (`run_tool_loop` in gemini_client.py).

Gemini itself is always mocked here via `_get_client` — these tests never make a live
call, they only verify the loop's own control flow: allowlisting, execution, multi-call
sequencing, and the hard tool-call ceiling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.gemini_client import ToolDeclaration, run_tool_loop


def _function_call(name: str, args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args or {})


def _model_turn(function_calls: list[SimpleNamespace] | None = None, text: str | None = None):
    """Builds a fake `response.candidates[0].content` turn with either function_call parts
    or a plain text part, matching the shape `run_tool_loop` reads."""
    if function_calls:
        parts = [SimpleNamespace(function_call=fc, text=None) for fc in function_calls]
    else:
        parts = [SimpleNamespace(function_call=None, text=text or "done")]
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def _client_with_responses(*responses) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = list(responses)
    return client


def _echo_tool(name: str, result="ok") -> ToolDeclaration:
    return ToolDeclaration(
        name=name,
        description=f"test tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        handler=lambda _args: result,
    )


class TestNoToolsDeclared:
    def test_returns_empty_without_calling_gemini(self):
        with patch("app.services.gemini_client._get_client") as mock_get_client:
            records = run_tool_loop(
                system_instruction="test", prompt="test", tools=[], max_tool_calls=5
            )

        assert records == []
        mock_get_client.assert_not_called()


class TestSingleToolCall:
    def test_executes_allowlisted_tool_and_returns_record(self):
        responses = [
            _model_turn([_function_call("get_employee")]),
            _model_turn(text="done"),
        ]
        with patch(
            "app.services.gemini_client._get_client",
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


class TestNoToolCallRequested:
    def test_returns_empty_when_model_never_calls_a_tool(self):
        with patch(
            "app.services.gemini_client._get_client",
            return_value=_client_with_responses(_model_turn(text="no tools needed")),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee")],
                max_tool_calls=5,
            )

        assert records == []


class TestMultipleToolCalls:
    def test_executes_several_tools_across_turns(self):
        responses = [
            _model_turn([_function_call("get_employee")]),
            _model_turn([_function_call("get_jira_activity")]),
            _model_turn(text="done"),
        ]
        with patch(
            "app.services.gemini_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee"), _echo_tool("get_jira_activity")],
                max_tool_calls=5,
            )

        assert [r.name for r in records] == ["get_employee", "get_jira_activity"]
        assert all(r.error is None for r in records)

    def test_executes_several_tools_requested_in_one_turn(self):
        responses = [
            _model_turn(
                [_function_call("get_employee"), _function_call("get_jira_activity")]
            ),
            _model_turn(text="done"),
        ]
        with patch(
            "app.services.gemini_client._get_client",
            return_value=_client_with_responses(*responses),
        ):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee"), _echo_tool("get_jira_activity")],
                max_tool_calls=5,
            )

        assert len(records) == 2
        assert {r.name for r in records} == {"get_employee", "get_jira_activity"}


class TestToolAllowlist:
    def test_unknown_tool_name_is_rejected_not_executed(self):
        handler = MagicMock()
        responses = [
            _model_turn([_function_call("delete_database")]),
            _model_turn(text="done"),
        ]
        with patch(
            "app.services.gemini_client._get_client",
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
        assert records[0].error is not None
        assert "not" in records[0].error.lower()


class TestMalformedToolCall:
    def test_handler_exception_is_captured_not_raised(self):
        def _boom(_args):
            raise ValueError("bad input")

        responses = [
            _model_turn([_function_call("get_employee")]),
            _model_turn(text="done"),
        ]
        with patch(
            "app.services.gemini_client._get_client",
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
        assert records[0].result is None


class TestMaxToolCallLimit:
    def test_stops_executing_once_limit_reached(self):
        # The model keeps requesting tool calls forever; the loop must still terminate
        # and must never execute more than max_tool_calls times.
        responses = [_model_turn([_function_call("get_employee")]) for _ in range(10)]
        client = _client_with_responses(*responses)

        with patch("app.services.gemini_client._get_client", return_value=client):
            records = run_tool_loop(
                system_instruction="test",
                prompt="test",
                tools=[_echo_tool("get_employee")],
                max_tool_calls=2,
            )

        executed = [r for r in records if r.error is None]
        assert len(executed) == 2
        # The loop must have terminated well before exhausting all 10 fake responses.
        assert client.models.generate_content.call_count <= 3

    def test_limit_of_zero_never_executes_anything(self):
        handler = MagicMock()
        with patch(
            "app.services.gemini_client._get_client",
            return_value=_client_with_responses(
                _model_turn([_function_call("get_employee")])
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
        assert len(records) == 0
