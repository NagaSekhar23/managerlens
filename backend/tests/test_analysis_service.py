from unittest.mock import patch

import pytest

import app.services.analysis_service as analysis_service_module
import app.services.groq_client as groq_client_module
from app.schemas.analysis import GeminiAnalysisPayload
from app.services.analysis_service import analyze_situation
from app.services.errors import KnowledgeRetrievalError, LLMRequestError
from app.services.retrieval_service import KnowledgeMatch

FAKE_PAYLOAD = GeminiAnalysisPayload(
    situation_summary="summary",
    situation_type="performance",
    confidence=0.5,
    observed_facts=["fact one"],
    assumptions_to_avoid=["assumption one"],
    missing_context=["missing one"],
    clarifying_questions=["question one"],
    recommended_actions=["action one"],
    conversation_plan=["step one"],
    risks=["risk one"],
    reasoning_basis="based only on the stated fact",
)

FAKE_MATCHES = [
    KnowledgeMatch(title="Performance Conversations", content="Distinguish skill from will.", score=0.71),
    KnowledgeMatch(title="One-on-Ones", content="Notice patterns across time.", score=0.63),
]


class TestSuccessfulRagAnalysis:
    def test_includes_retrieved_knowledge_and_prompt_evidence(self):
        with (
            patch(
                "app.services.analysis_service.retrieve_relevant_knowledge",
                return_value=FAKE_MATCHES,
            ),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("My engineer missed a deadline.")

        assert result.situation_summary == FAKE_PAYLOAD.situation_summary
        assert len(result.knowledge_used) == 2
        assert result.knowledge_used[0].title == "Performance Conversations"
        assert result.knowledge_used[0].relevance_score == 0.71

        _, kwargs = mock_generate.call_args
        assert "My engineer missed a deadline." in kwargs["prompt"]
        assert "Performance Conversations" in kwargs["prompt"]
        assert kwargs["response_model"] is GeminiAnalysisPayload
        assert "never invent facts" in kwargs["system_instruction"].lower()
        assert "never treat it as a" in kwargs["system_instruction"].lower()

    def test_no_relevant_knowledge_still_produces_analysis(self):
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Vague situation with no clear match.")

        assert result.knowledge_used == []
        _, kwargs = mock_generate.call_args
        assert "no relevant guidance was found" in kwargs["prompt"].lower()


class TestGeminiFailure:
    def test_generation_failure_propagates(self):
        with (
            patch(
                "app.services.analysis_service.retrieve_relevant_knowledge",
                return_value=FAKE_MATCHES,
            ),
            patch(
                "app.services.analysis_service.generate_structured",
                side_effect=LLMRequestError("Gemini request failed: timeout"),
            ),
        ):
            with pytest.raises(LLMRequestError):
                analyze_situation("My engineer missed a deadline.")


class TestKnowledgeBaseFailure:
    def test_retrieval_failure_degrades_gracefully(self):
        with (
            patch(
                "app.services.analysis_service.retrieve_relevant_knowledge",
                side_effect=KnowledgeRetrievalError("Knowledge base query failed: connection refused"),
            ),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("My engineer missed a deadline.")

        assert result.knowledge_used == []
        _, kwargs = mock_generate.call_args
        assert "no relevant guidance was found" in kwargs["prompt"].lower()


class TestUsesGroqForReasoning:
    """Confirms analyze_situation's reasoning/tool-calling calls are wired to the Groq
    client, not Gemini — this is the actual provider swap this migration makes."""

    def test_generate_structured_and_run_tool_loop_come_from_groq_client(self):
        assert analysis_service_module.generate_structured is groq_client_module.generate_structured
        assert analysis_service_module.run_tool_loop is groq_client_module.run_tool_loop
