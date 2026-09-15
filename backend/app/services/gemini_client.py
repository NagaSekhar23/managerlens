"""Thin, provider-specific abstraction over the Gemini API.

Callers only ever import `generate_text` / `generate_structured` from this
module — nothing else in the app touches the `google.genai` SDK directly.
That keeps a future provider swap to a single file, and keeps the API key
out of every layer above this one.
"""

from typing import TypeVar

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
