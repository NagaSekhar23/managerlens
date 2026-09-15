import pytest

from app.config import settings
from app.services.gemini_client import generate_text

requires_gemini_key = pytest.mark.skipif(
    not settings.gemini_api_key,
    reason="GEMINI_API_KEY is not set in the project root .env",
)


@requires_gemini_key
def test_generate_text_returns_nonempty_reply():
    reply = generate_text("Reply with exactly one word: pong")

    assert isinstance(reply, str)
    assert reply.strip() != ""
