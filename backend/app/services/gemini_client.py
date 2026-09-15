"""Thin, provider-specific abstraction over the Gemini API.

Callers only ever import `generate_text` / `generate_structured` / `run_tool_loop`
from this module — nothing else in the app touches the `google.genai` SDK
directly. That keeps a future provider swap to a single file, and keeps the
API key out of every layer above this one.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.services.embedding_config import EMBEDDING_DIMENSIONS
from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError

ModelT = TypeVar("ModelT", bound=BaseModel)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not configured. Set it in the project root .env file."
        )

    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.gemini_timeout_seconds * 1000)
            ),
        )
    return _client


def generate_text(prompt: str) -> str:
    """Send a single prompt to Gemini and return the model's plain-text reply.

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: the request to Gemini failed (network, auth, rate limit, etc.).
    """
    client = _get_client()

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
    except APIError as exc:
        raise LLMRequestError(f"Gemini request failed: {exc}") from exc
    except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
        raise LLMRequestError(f"Gemini request failed: {exc}") from exc

    if not response.text:
        raise LLMRequestError("Gemini returned an empty response.")

    return response.text


def generate_structured(
    *,
    system_instruction: str,
    prompt: str,
    response_model: type[ModelT],
) -> ModelT:
    """Send a prompt to Gemini and parse its reply into `response_model`.

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: the request to Gemini failed (network, auth, rate limit, etc.).
        InvalidLLMOutputError: the response could not be parsed into the schema.
    """
    client = _get_client()

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
    except APIError as exc:
        raise LLMRequestError(f"Gemini request failed: {exc}") from exc
    except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
        raise LLMRequestError(f"Gemini request failed: {exc}") from exc

    parsed = response.parsed
    if parsed is None:
        raise InvalidLLMOutputError(
            "Gemini did not return output matching the expected structured schema."
        )

    try:
        return response_model.model_validate(parsed, from_attributes=True)
    except ValidationError as exc:
        raise InvalidLLMOutputError(f"Gemini response failed schema validation: {exc}") from exc


def embed_text(text: str, *, task_type: str) -> list[float]:
    """Embed a piece of text with Gemini's embedding model.

    `task_type` should be "RETRIEVAL_DOCUMENT" when embedding knowledge-base
    chunks at ingestion time, and "RETRIEVAL_QUERY" when embedding a manager's
    situation at query time — Gemini optimizes the embedding differently for
    each role, which meaningfully improves retrieval quality.

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: the request to Gemini failed, or returned no embedding.
    """
    client = _get_client()

    try:
        response = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
                task_type=task_type,
            ),
        )
    except APIError as exc:
        raise LLMRequestError(f"Gemini embedding request failed: {exc}") from exc
    except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
        raise LLMRequestError(f"Gemini embedding request failed: {exc}") from exc

    if not response.embeddings:
        raise LLMRequestError("Gemini returned no embedding.")

    return list(response.embeddings[0].values)


@dataclass(frozen=True)
class ToolDeclaration:
    """One tool Gemini is allowed to call during a `run_tool_loop`.

    `handler` receives the raw arguments dict Gemini supplied (empty if the tool takes no
    parameters) and must return JSON-serializable data (a dict, list, or primitive) — that
    return value is sent straight back to Gemini as the function's result.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolCallRecord:
    """What actually happened for one executed (or rejected) tool call."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


def run_tool_loop(
    *,
    system_instruction: str,
    prompt: str,
    tools: list[ToolDeclaration],
    max_tool_calls: int,
) -> list[ToolCallRecord]:
    """Drive a bounded Gemini function-calling conversation and return a record of every
    tool call that was actually executed (or rejected).

    The loop:
    1. Gives Gemini only the declared `tools`.
    2. Lets Gemini request a tool call.
    3. Executes it ONLY if its name is in the `tools` allowlist — anything else is rejected
       with an error response and never runs.
    4. Sends the tool's result (or the rejection) back to Gemini as a function response.
    5. Lets Gemini continue reasoning, possibly requesting further tool calls.
    6. Hard-stops once `max_tool_calls` tools have been executed, regardless of what Gemini
       asks for afterward — no exceptions.

    This function never executes arbitrary code: `tools` is a fixed, closed list of Python
    callables chosen by our own code, not by Gemini. Gemini can only choose *which* of those
    fixed callables to invoke and *how many times*, never *what code runs*.

    Returns an empty list immediately if `tools` is empty (nothing to call).

    Raises:
        MissingAPIKeyError: no API key configured.
        LLMRequestError: a request to Gemini failed (network, auth, rate limit, etc.).
    """
    if not tools:
        return []

    client = _get_client()
    allowlist = {tool.name: tool for tool in tools}

    gemini_tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters_schema,
                )
                for tool in tools
            ]
        )
    ]

    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    records: list[ToolCallRecord] = []
    calls_made = 0

    # Bounded by construction: each pass through this loop either makes zero tool calls (and
    # breaks immediately) or makes at least one (advancing calls_made toward max_tool_calls),
    # so `max_tool_calls + 1` iterations is always enough — this cannot spin forever.
    for _ in range(max_tool_calls + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=gemini_tools,
                ),
            )
        except APIError as exc:
            raise LLMRequestError(f"Gemini request failed: {exc}") from exc
        except Exception as exc:  # network errors, timeouts, etc. from the SDK/httpx layer
            raise LLMRequestError(f"Gemini request failed: {exc}") from exc

        candidates = response.candidates or []
        if not candidates:
            break

        model_content = candidates[0].content
        parts = model_content.parts or []
        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not function_calls:
            break

        contents.append(model_content)

        response_parts: list[types.Part] = []
        limit_reached = False

        for call in function_calls:
            arguments = dict(call.args or {})

            if calls_made >= max_tool_calls:
                limit_reached = True
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={
                            "error": "Tool call limit reached; no further tools were executed."
                        },
                    )
                )
                continue

            tool = allowlist.get(call.name)
            if tool is None:
                records.append(
                    ToolCallRecord(
                        name=call.name,
                        arguments=arguments,
                        error=f"'{call.name}' is not an allowlisted tool; it was not executed.",
                    )
                )
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"error": f"Unknown tool '{call.name}'; it was not executed."},
                    )
                )
                continue

            calls_made += 1
            try:
                result = tool.handler(arguments)
            except Exception as exc:
                records.append(
                    ToolCallRecord(name=call.name, arguments=arguments, error=str(exc))
                )
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name, response={"error": str(exc)}
                    )
                )
                continue

            records.append(ToolCallRecord(name=call.name, arguments=arguments, result=result))
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))

        if limit_reached:
            break

    return records
