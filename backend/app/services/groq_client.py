"""Thin, provider-specific abstraction over the Groq API for reasoning, tool
calling, and structured output.

Groq is the reasoning/tool-calling provider — embeddings stay on Gemini
(see gemini_client.embed_text), since Groq has no embeddings API and the
existing pgvector knowledge base was built with Gemini's embeddings.

Groq (like most OpenAI-compatible chat APIs) does not reliably support tool
calling and strict structured JSON output in the same request, so this module
keeps them as two separate calls, mirroring exactly how gemini_client already
splits `run_tool_loop` (tool calling only) from `generate_structured`
(structured output only, no tools) — analysis_service.py already calls them
as two separate phases, so no change was needed to that call order.
"""

import json
from typing import TypeVar

from groq import APIError, Groq
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError
from app.services.gemini_client import ToolCallRecord, ToolDeclaration

ModelT = TypeVar("ModelT", bound=BaseModel)

_client: Groq | None = None


def _get_client() -> Groq:
    if not settings.groq_api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not configured. Set it in the project root .env file."
        )

    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout_seconds)
    return _client


def generate_structured(
    *,
    system_instruction: str,
    prompt: str,
    response_model: type[ModelT],
) -> ModelT:
    """Send a prompt to Groq (no tools) and parse its reply into `response_model`.

    This is always called AFTER any tool calling is done (see `run_tool_loop`) — Groq's
    JSON-schema structured output is requested on its own, tool-free request, the same
    two-phase split gemini_client already used.

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: the request to Groq failed (network, auth, rate limit, etc.).
        InvalidLLMOutputError: the response could not be parsed into the schema.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        )
    except APIError as exc:
        raise LLMRequestError(f"Groq request failed: {exc}") from exc
    except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
        raise LLMRequestError(f"Groq request failed: {exc}") from exc

    choices = response.choices or []
    content = choices[0].message.content if choices and choices[0].message else None
    if not content:
        raise InvalidLLMOutputError(
            "Groq did not return output matching the expected structured schema."
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidLLMOutputError(f"Groq response was not valid JSON: {exc}") from exc

    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise InvalidLLMOutputError(f"Groq response failed schema validation: {exc}") from exc


def run_tool_loop(
    *,
    system_instruction: str,
    prompt: str,
    tools: list[ToolDeclaration],
    max_tool_calls: int,
) -> list[ToolCallRecord]:
    """Drive a bounded Groq function-calling conversation and return a record of every
    tool call that was actually executed (or rejected).

    Behavior is identical to `gemini_client.run_tool_loop` (see that docstring for the
    full safety rundown — allowlist enforcement, hard `max_tool_calls` ceiling, no
    arbitrary code execution, employee identity fixed by closure, never by the model):
    only the request/response shapes differ because Groq's chat-completions API is
    OpenAI-style (`tool_calls` on the assistant message, `role: "tool"` reply messages
    keyed by `tool_call_id`) rather than Gemini's `function_call`/`function_response`
    parts.

    Returns an empty list immediately if `tools` is empty (nothing to call).

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: a request to Groq failed (network, auth, rate limit, etc.).
    """
    if not tools:
        return []

    client = _get_client()
    allowlist = {tool.name: tool for tool in tools}

    groq_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }
        for tool in tools
    ]

    messages: list[dict] = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    records: list[ToolCallRecord] = []
    calls_made = 0

    # Bounded by construction: each pass through this loop either makes zero tool calls (and
    # breaks immediately) or makes at least one (advancing calls_made toward max_tool_calls),
    # so `max_tool_calls + 1` iterations is always enough — this cannot spin forever.
    for _ in range(max_tool_calls + 1):
        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                tools=groq_tools,
            )
        except APIError as exc:
            raise LLMRequestError(f"Groq request failed: {exc}") from exc
        except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
            raise LLMRequestError(f"Groq request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            break

        message = choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        limit_reached = False

        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments) if call.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {}

            if calls_made >= max_tool_calls:
                limit_reached = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"error": "Tool call limit reached; no further tools were executed."}
                        ),
                    }
                )
                continue

            tool = allowlist.get(call.function.name)
            if tool is None:
                records.append(
                    ToolCallRecord(
                        name=call.function.name,
                        arguments=arguments,
                        error=f"'{call.function.name}' is not an allowlisted tool; it was not executed.",
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"error": f"Unknown tool '{call.function.name}'; it was not executed."}
                        ),
                    }
                )
                continue

            calls_made += 1
            try:
                result = tool.handler(arguments)
            except Exception as exc:
                records.append(
                    ToolCallRecord(name=call.function.name, arguments=arguments, error=str(exc))
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": str(exc)}),
                    }
                )
                continue

            records.append(ToolCallRecord(name=call.function.name, arguments=arguments, result=result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"result": result}),
                }
            )

        if limit_reached:
            break

    return records
