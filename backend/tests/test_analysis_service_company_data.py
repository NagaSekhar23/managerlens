"""Tests for the company-data tool wiring inside analyze_situation — employee
identification gating, evidence prompting, company_data_used, and graceful
degradation. Gemini is always mocked; existing RAG-only behavior is covered
separately in test_analysis_service.py and must stay unaffected by any of this
(verified again here in TestExistingBehaviorUnaffected).
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.analysis import GeminiAnalysisPayload
from app.services.analysis_service import analyze_situation
from app.services.errors import LLMRequestError
from app.services.gemini_client import ToolCallRecord

JORDAN = SimpleNamespace(
    employee_id="EMP001",
    name="Jordan Alvarez",
    role="Software Engineer",
    team="Platform",
    manager="Sam Rivera",
    start_date="2024-02-12",
)
CASEY = SimpleNamespace(
    employee_id="EMP003",
    name="Casey Kim",
    role="Software Engineer",
    team="Platform",
    manager="Sam Rivera",
    start_date="2026-08-01",
)

FAKE_PAYLOAD = GeminiAnalysisPayload(
    situation_summary="summary",
    situation_type="performance",
    confidence=0.5,
    observed_facts=["fact one"],
    assumptions_to_avoid=[],
    missing_context=[],
    clarifying_questions=[],
    recommended_actions=[],
    conversation_plan=[],
    risks=[],
    reasoning_basis="based only on the stated fact",
)


class TestEmployeeIdentifiedIncludesCompanyEvidence:
    def test_company_evidence_appears_in_prompt_and_result(self):
        jira_issue = {
            "employee_id": "EMP001",
            "issue_id": "PLAT-101",
            "title": "Implement rate limiter",
            "status": "Done",
            "due_date": "2026-06-15",
            "completed_date": "2026-06-28",
            "blocked_by": None,
            "project": "Platform",
        }
        records = [
            ToolCallRecord(name="get_jira_activity", arguments={}, result=[jira_issue]),
            ToolCallRecord(
                name="get_employee",
                arguments={},
                result={
                    "employee_id": "EMP001",
                    "name": "Jordan Alvarez",
                    "role": "Software Engineer",
                    "team": "Platform",
                    "manager": "Sam Rivera",
                    "start_date": "2024-02-12",
                },
            ),
        ]
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text",
                return_value="EMP001",
            ),
            patch("app.services.analysis_service.get_employee", return_value=JORDAN),
            patch(
                "app.services.analysis_service.run_tool_loop", return_value=records
            ) as mock_run_tool_loop,
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Jordan Alvarez missed two deadlines.")

        mock_run_tool_loop.assert_called_once()
        _, tool_kwargs = mock_run_tool_loop.call_args
        assert len(tool_kwargs["tools"]) == 5
        assert {t.name for t in tool_kwargs["tools"]} == {
            "get_employee",
            "get_jira_activity",
            "get_github_activity",
            "get_one_on_ones",
            "get_feedback_history",
        }

        _, gen_kwargs = mock_generate.call_args
        assert "Observed company evidence for employee EMP001" in gen_kwargs["prompt"]
        assert "PLAT-101" in gen_kwargs["prompt"]

        assert len(result.company_data_used) == 2
        tool_names = {c.tool for c in result.company_data_used}
        assert tool_names == {"get_jira_activity", "get_employee"}
        assert all(c.employee_id == "EMP001" for c in result.company_data_used)
        assert all(c.employee_name == "Jordan Alvarez" for c in result.company_data_used)
        jira_source = next(c for c in result.company_data_used if c.tool == "get_jira_activity")
        assert jira_source.summary.startswith("1 record(s) found.")
        assert "PLAT-101" in jira_source.summary
        assert "Implement rate limiter" in jira_source.summary
        assert "completed 2026-06-28" in jira_source.summary


class TestNoEmployeeIdentified:
    def test_company_tool_loop_is_never_invoked(self):
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text", return_value=None
            ),
            patch("app.services.analysis_service.run_tool_loop") as mock_run_tool_loop,
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ),
        ):
            result = analyze_situation("My engineer missed a deadline.")

        mock_run_tool_loop.assert_not_called()
        assert result.company_data_used == []


class TestSparseEmployeeEvidence:
    def test_empty_tool_results_are_reported_not_invented(self):
        records = [ToolCallRecord(name="get_one_on_ones", arguments={}, result=[])]
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text",
                return_value="EMP003",
            ),
            patch("app.services.analysis_service.get_employee", return_value=CASEY),
            patch("app.services.analysis_service.run_tool_loop", return_value=records),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Casey Kim just joined, how are they doing?")

        _, gen_kwargs = mock_generate.call_args
        assert "no records found" in gen_kwargs["prompt"].lower()
        assert len(result.company_data_used) == 1
        assert result.company_data_used[0].summary == "No records found."

    def test_no_tools_called_is_reported_as_absence_not_invented(self):
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text",
                return_value="EMP003",
            ),
            patch("app.services.analysis_service.get_employee", return_value=CASEY),
            patch("app.services.analysis_service.run_tool_loop", return_value=[]),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Casey Kim just joined, how are they doing?")

        _, gen_kwargs = mock_generate.call_args
        assert "no company-data tools were called for employee emp003" in gen_kwargs["prompt"].lower()
        assert result.company_data_used == []


class TestToolLoopFailureDegradesGracefully:
    def test_llm_request_error_from_tool_loop_does_not_fail_the_request(self):
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text",
                return_value="EMP001",
            ),
            patch("app.services.analysis_service.get_employee", return_value=JORDAN),
            patch(
                "app.services.analysis_service.run_tool_loop",
                side_effect=LLMRequestError("Gemini request failed: timeout"),
            ),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Jordan Alvarez missed a deadline.")

        assert result.company_data_used == []
        _, gen_kwargs = mock_generate.call_args
        assert "no company-data tools were called" in gen_kwargs["prompt"].lower()


class TestErroredToolCallsExcludedFromCompanyDataUsed:
    def test_failed_lookup_is_shown_in_prompt_but_not_surfaced_as_used_evidence(self):
        records = [
            ToolCallRecord(name="get_jira_activity", arguments={}, error="boom"),
            ToolCallRecord(name="get_employee", arguments={}, result={"name": "Jordan Alvarez"}),
        ]
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch(
                "app.services.analysis_service.find_employee_id_in_text",
                return_value="EMP001",
            ),
            patch("app.services.analysis_service.get_employee", return_value=JORDAN),
            patch("app.services.analysis_service.run_tool_loop", return_value=records),
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Jordan Alvarez missed a deadline.")

        _, gen_kwargs = mock_generate.call_args
        assert "lookup failed" in gen_kwargs["prompt"].lower()
        assert len(result.company_data_used) == 1
        assert result.company_data_used[0].tool == "get_employee"


class TestSummarizeCompanyRecord:
    """Direct tests for `_summarize_company_record` — the function that turns raw tool
    output into the concise, non-fabricated text the frontend shows as company evidence."""

    def test_get_employee_summary(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_employee",
            {
                "employee_id": "EMP001",
                "name": "Jordan Alvarez",
                "role": "Software Engineer",
                "team": "Platform",
                "manager": "Sam Rivera",
                "start_date": "2024-02-12",
            },
        )
        assert "Software Engineer" in summary
        assert "Platform" in summary
        assert "Sam Rivera" in summary
        assert "2024-02-12" in summary

    def test_get_jira_activity_summary_uses_completed_date_when_done(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_jira_activity",
            [
                {
                    "issue_id": "PLAT-101",
                    "title": "Rate limiter",
                    "status": "Done",
                    "due_date": "2026-06-15",
                    "completed_date": "2026-06-28",
                }
            ],
        )
        assert "PLAT-101" in summary
        assert "completed 2026-06-28" in summary
        assert "due 2026-06-15" not in summary

    def test_get_jira_activity_summary_uses_due_date_when_not_completed(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_jira_activity",
            [{"issue_id": "PLAT-142", "title": "Flaky tests", "status": "In Progress",
              "due_date": "2026-09-20", "completed_date": None}],
        )
        assert "due 2026-09-20" in summary

    def test_get_github_activity_summary(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_github_activity",
            [{"pull_request": 482, "title": "Add rate limiter middleware", "merged_at": "2026-06-27"}],
        )
        assert "PR #482" in summary
        assert "merged 2026-06-27" in summary

    def test_get_one_on_ones_summary_includes_manager_notes(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_one_on_ones",
            [{"date": "2026-06-20", "manager_notes": "The delay looks external, not effort-related."}],
        )
        assert "2026-06-20" in summary
        assert "external" in summary

    def test_get_feedback_history_summary(self):
        from app.services.analysis_service import _summarize_company_record

        summary = _summarize_company_record(
            "get_feedback_history",
            [{"date": "2026-08-05", "feedback_type": "manager", "summary": "Work quality remains high."}],
        )
        assert "manager" in summary
        assert "Work quality remains high." in summary

    def test_empty_list_is_reported_plainly(self):
        from app.services.analysis_service import _summarize_company_record

        assert _summarize_company_record("get_jira_activity", []) == "No records found."

    def test_long_summary_is_truncated(self):
        from app.services.analysis_service import _summarize_company_record

        many_notes = [{"date": f"2026-01-{i:02d}", "manager_notes": "x" * 50} for i in range(1, 10)]
        summary = _summarize_company_record("get_one_on_ones", many_notes)
        assert len(summary) <= 281  # COMPANY_SUMMARY_LENGTH + ellipsis
        assert summary.endswith("…")


class TestExistingBehaviorUnaffected:
    def test_generic_situation_without_named_employee_matches_prior_rag_only_behavior(self):
        with (
            patch("app.services.analysis_service.retrieve_relevant_knowledge", return_value=[]),
            patch("app.services.analysis_service.run_tool_loop") as mock_run_tool_loop,
            patch(
                "app.services.analysis_service.generate_structured", return_value=FAKE_PAYLOAD
            ) as mock_generate,
        ):
            result = analyze_situation("Vague situation with no clear match.")

        mock_run_tool_loop.assert_not_called()
        assert result.knowledge_used == []
        assert result.company_data_used == []
        _, gen_kwargs = mock_generate.call_args
        assert "no relevant guidance was found" in gen_kwargs["prompt"].lower()
