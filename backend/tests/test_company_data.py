"""Validates backend/data/company/ — the mock (simulated) company dataset used
for local demos and evaluation. No application code reads this data yet
(Phase 1 only creates it); these tests just guard the data's own integrity
so a later tool-building phase can trust its shape.
"""

import json
import re
from pathlib import Path

import pytest

COMPANY_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "company"

EMPLOYEE_REQUIRED_FIELDS = {"employee_id", "name", "role", "team", "manager", "start_date"}
JIRA_REQUIRED_FIELDS = {
    "employee_id",
    "issue_id",
    "title",
    "status",
    "due_date",
    "completed_date",
    "blocked_by",
    "project",
}
GITHUB_REQUIRED_FIELDS = {
    "employee_id",
    "repository",
    "pull_request",
    "title",
    "opened_at",
    "merged_at",
    "review_count",
}
ONE_ON_ONE_REQUIRED_FIELDS = {"employee_id", "date", "topics", "concerns", "manager_notes"}
FEEDBACK_REQUIRED_FIELDS = {"employee_id", "date", "feedback_type", "summary"}

# Deliberately simple, deterministic PII heuristics — this is synthetic data we wrote
# ourselves, so a plain pattern check is enough to guard against ever accidentally
# pasting in something real (an email, phone number, or SSN-shaped string).
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _load(filename: str):
    path = COMPANY_DATA_DIR / filename
    with path.open() as f:
        return json.load(f)


def _all_text(data) -> str:
    return json.dumps(data)


@pytest.fixture(scope="module")
def employees():
    return _load("employees.json")


@pytest.fixture(scope="module")
def jira_activity():
    return _load("jira_activity.json")


@pytest.fixture(scope="module")
def github_activity():
    return _load("github_activity.json")


@pytest.fixture(scope="module")
def one_on_ones():
    return _load("one_on_ones.json")


@pytest.fixture(scope="module")
def feedback_history():
    return _load("feedback_history.json")


class TestFilesAreValidJson:
    @pytest.mark.parametrize(
        "filename",
        [
            "employees.json",
            "jira_activity.json",
            "github_activity.json",
            "one_on_ones.json",
            "feedback_history.json",
        ],
    )
    def test_file_loads_as_json(self, filename):
        data = _load(filename)
        assert isinstance(data, list)
        assert len(data) > 0


class TestRequiredFields:
    def test_employees_have_required_fields(self, employees):
        for record in employees:
            assert EMPLOYEE_REQUIRED_FIELDS.issubset(record.keys())

    def test_jira_activity_has_required_fields(self, jira_activity):
        for record in jira_activity:
            assert JIRA_REQUIRED_FIELDS.issubset(record.keys())

    def test_github_activity_has_required_fields(self, github_activity):
        for record in github_activity:
            assert GITHUB_REQUIRED_FIELDS.issubset(record.keys())

    def test_one_on_ones_have_required_fields(self, one_on_ones):
        for record in one_on_ones:
            assert ONE_ON_ONE_REQUIRED_FIELDS.issubset(record.keys())

    def test_feedback_history_has_required_fields(self, feedback_history):
        for record in feedback_history:
            assert FEEDBACK_REQUIRED_FIELDS.issubset(record.keys())


class TestReferentialIntegrity:
    def test_employee_ids_are_unique(self, employees):
        ids = [e["employee_id"] for e in employees]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "fixture_name",
        ["jira_activity", "github_activity", "one_on_ones", "feedback_history"],
    )
    def test_activity_employee_ids_exist_in_employees(self, request, employees, fixture_name):
        known_ids = {e["employee_id"] for e in employees}
        activity = request.getfixturevalue(fixture_name)
        referenced_ids = {record["employee_id"] for record in activity}
        unknown = referenced_ids - known_ids
        assert not unknown, f"{fixture_name} references unknown employee_id(s): {unknown}"

    def test_every_employee_has_at_least_one_jira_issue(self, employees, jira_activity):
        known_ids = {e["employee_id"] for e in employees}
        covered_ids = {record["employee_id"] for record in jira_activity}
        assert known_ids == covered_ids


class TestNoRealPersonalInformation:
    @pytest.mark.parametrize(
        "filename",
        [
            "employees.json",
            "jira_activity.json",
            "github_activity.json",
            "one_on_ones.json",
            "feedback_history.json",
        ],
    )
    def test_no_email_phone_or_ssn_patterns(self, filename):
        data = _load(filename)
        text = _all_text(data)
        assert not EMAIL_PATTERN.search(text), f"{filename} contains an email-like string"
        assert not PHONE_PATTERN.search(text), f"{filename} contains a phone-number-like string"
        assert not SSN_PATTERN.search(text), f"{filename} contains an SSN-like string"


class TestNarrativeCoverage:
    """Confirms the dataset actually demonstrates the three required cases —
    not just that the files are well-formed."""

    def test_one_employee_has_external_blockers_despite_missed_deadlines(
        self, jira_activity, feedback_history
    ):
        # EMP001: at least one Jira issue completed after its due date, with a
        # non-null blocked_by explanation.
        late_with_blocker = [
            r
            for r in jira_activity
            if r["employee_id"] == "EMP001"
            and r["completed_date"]
            and r["due_date"]
            and r["completed_date"] > r["due_date"]
            and r["blocked_by"]
        ]
        assert len(late_with_blocker) >= 2

        # And feedback history eventually attributes the delay to something external.
        later_feedback = " ".join(
            f["summary"] for f in feedback_history if f["employee_id"] == "EMP001"
        ).lower()
        assert "external" in later_feedback or "blocked" in later_feedback or "blocker" in later_feedback

    def test_one_employee_has_consistently_strong_signals(self, jira_activity, feedback_history):
        # EMP002: every completed Jira issue finished on or before its due date.
        emp2_issues = [r for r in jira_activity if r["employee_id"] == "EMP002" and r["completed_date"]]
        assert len(emp2_issues) >= 2
        assert all(r["completed_date"] <= r["due_date"] for r in emp2_issues)

        emp2_feedback = feedback_history and [
            f for f in feedback_history if f["employee_id"] == "EMP002"
        ]
        assert len(emp2_feedback) >= 2

    def test_one_employee_has_sparse_ambiguous_history(
        self, jira_activity, github_activity, one_on_ones
    ):
        # EMP003: deliberately thin evidence trail across every source.
        assert len([r for r in jira_activity if r["employee_id"] == "EMP003"]) <= 3
        assert len([r for r in github_activity if r["employee_id"] == "EMP003"]) <= 2
        assert len([r for r in one_on_ones if r["employee_id"] == "EMP003"]) <= 1
