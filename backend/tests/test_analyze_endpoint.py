from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.analysis import MAX_SITUATION_LENGTH, AnalysisResult
from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError

client = TestClient(app)

VALID_SITUATION = (
    "My engineer has missed three deadlines and has become quiet during meetings. "
    "I am worried they are disengaged."
)


def _sample_result() -> AnalysisResult:
    return AnalysisResult(
        situation_summary="An engineer missed three deadlines and has been quieter in meetings.",
        situation_type="performance",
        confidence=0.4,
        observed_facts=[
            "The engineer missed three deadlines.",
            "The engineer has become quieter during team meetings.",
        ],
        assumptions_to_avoid=[
            "That the engineer is disengaged or no longer cares about their work.",
        ],
        missing_context=[
            "Whether the deadlines were realistic given the engineer's workload.",
            "Whether anything has changed in the engineer's personal life or team dynamics.",
        ],
        clarifying_questions=[
            "What did the engineer say, if anything, about the missed deadlines?",
            "Has their workload or team situation changed recently?",
        ],
        recommended_actions=[
            "Schedule a private 1:1 to check in before drawing conclusions.",
        ],
        conversation_plan=[
            "Open by asking how they're doing, not by listing the missed deadlines.",
            "Share the observed pattern factually and ask for their perspective.",
        ],
        risks=[
            "Acting on the assumption of disengagement could damage trust if it's inaccurate.",
        ],
        reasoning_basis=(
            "Based only on the two facts stated; confidence is low because motive and context "
            "are unknown."
        ),
    )


class TestRequestValidation:
    def test_empty_situation_is_rejected(self):
        response = client.post("/api/analyze", json={"situation": ""})
        assert response.status_code == 422

    def test_whitespace_only_situation_is_rejected(self):
        response = client.post("/api/analyze", json={"situation": "   "})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_missing_situation_field_is_rejected(self):
        response = client.post("/api/analyze", json={})
        assert response.status_code == 422

    def test_excessively_long_situation_is_rejected(self):
        too_long = "a" * (MAX_SITUATION_LENGTH + 1)
        response = client.post("/api/analyze", json={"situation": too_long})
        assert response.status_code == 422


class TestSuccessfulAnalysis:
    def test_returns_structured_analysis(self):
        with patch(
            "app.routers.analyze.analyze_situation", return_value=_sample_result()
        ):
            response = client.post("/api/analyze", json={"situation": VALID_SITUATION})

        assert response.status_code == 200
        body = response.json()
        assert body["situation_type"] == "performance"
        assert body["observed_facts"] == _sample_result().observed_facts
        assert body["assumptions_to_avoid"] == _sample_result().assumptions_to_avoid
        assert 0.0 <= body["confidence"] <= 1.0


class TestFailureModes:
    def test_missing_api_key_returns_500(self):
        with patch(
            "app.routers.analyze.analyze_situation",
            side_effect=MissingAPIKeyError("GEMINI_API_KEY is not configured."),
        ):
            response = client.post("/api/analyze", json={"situation": VALID_SITUATION})

        assert response.status_code == 500
        assert "not configured" in response.json()["detail"].lower()

    def test_gemini_request_failure_returns_502(self):
        with patch(
            "app.routers.analyze.analyze_situation",
            side_effect=LLMRequestError("Gemini request failed: network timeout"),
        ):
            response = client.post("/api/analyze", json={"situation": VALID_SITUATION})

        assert response.status_code == 502
        assert "temporarily unavailable" in response.json()["detail"].lower()

    def test_malformed_gemini_output_returns_502(self):
        with patch(
            "app.routers.analyze.analyze_situation",
            side_effect=InvalidLLMOutputError("Gemini response failed schema validation"),
        ):
            response = client.post("/api/analyze", json={"situation": VALID_SITUATION})

        assert response.status_code == 502
        assert "unexpected response" in response.json()["detail"].lower()
