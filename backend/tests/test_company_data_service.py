import pytest

from app.services.company_data_service import (
    find_employee_id_in_text,
    get_employee,
    get_feedback_history,
    get_github_activity,
    get_jira_activity,
    get_one_on_ones,
)
from app.services.errors import EmployeeNotFoundError


class TestGetEmployee:
    def test_returns_known_employee(self):
        employee = get_employee("EMP001")
        assert employee.employee_id == "EMP001"
        assert employee.name == "Jordan Alvarez"
        assert employee.manager == "Sam Rivera"

    def test_unknown_employee_raises(self):
        with pytest.raises(EmployeeNotFoundError):
            get_employee("EMP999")


class TestGetJiraActivity:
    def test_returns_only_that_employees_issues(self):
        issues = get_jira_activity("EMP001")
        assert len(issues) > 0
        assert all(i.employee_id == "EMP001" for i in issues)

    def test_unknown_employee_raises(self):
        with pytest.raises(EmployeeNotFoundError):
            get_jira_activity("EMP999")

    def test_sparse_employee_returns_short_list(self):
        # EMP003 (Casey Kim) is the deliberately sparse recent-hire case.
        issues = get_jira_activity("EMP003")
        assert 0 < len(issues) <= 3


class TestGetGithubActivity:
    def test_returns_only_that_employees_prs(self):
        prs = get_github_activity("EMP002")
        assert len(prs) > 0
        assert all(p.employee_id == "EMP002" for p in prs)

    def test_unknown_employee_raises(self):
        with pytest.raises(EmployeeNotFoundError):
            get_github_activity("EMP999")

    def test_sparse_employee_returns_short_list(self):
        prs = get_github_activity("EMP003")
        assert len(prs) <= 2


class TestGetOneOnOnes:
    def test_returns_only_that_employees_notes(self):
        notes = get_one_on_ones("EMP001")
        assert len(notes) > 0
        assert all(n.employee_id == "EMP001" for n in notes)

    def test_unknown_employee_raises(self):
        with pytest.raises(EmployeeNotFoundError):
            get_one_on_ones("EMP999")

    def test_sparse_employee_has_minimal_notes(self):
        notes = get_one_on_ones("EMP003")
        assert len(notes) <= 1


class TestGetFeedbackHistory:
    def test_returns_only_that_employees_feedback(self):
        feedback = get_feedback_history("EMP002")
        assert len(feedback) > 0
        assert all(f.employee_id == "EMP002" for f in feedback)

    def test_unknown_employee_raises(self):
        with pytest.raises(EmployeeNotFoundError):
            get_feedback_history("EMP999")

    def test_known_employee_with_no_matching_feedback_returns_empty_list(self, monkeypatch):
        # Exercise the "employee exists, but this particular source has nothing" path
        # without depending on the dataset never adding EMP003 feedback later.
        monkeypatch.setattr(
            "app.services.company_data_service._feedback_history",
            lambda: [],
        )
        assert get_feedback_history("EMP001") == []


class TestFindEmployeeIdInText:
    def test_matches_by_full_name(self):
        assert find_employee_id_in_text("Jordan Alvarez missed a deadline again.") == "EMP001"

    def test_matches_by_employee_id(self):
        assert find_employee_id_in_text("Can you look into EMP002's recent work?") == "EMP002"

    def test_is_case_insensitive(self):
        assert find_employee_id_in_text("i'm worried about priya natarajan") == "EMP002"

    def test_returns_none_when_no_employee_named(self):
        assert find_employee_id_in_text("My engineer missed a deadline.") is None

    def test_returns_none_for_unrecognized_id_shaped_text(self):
        # An id-shaped mention that doesn't correspond to any real employee should be
        # treated the same as "no employee named" — never guessed at.
        assert find_employee_id_in_text("EMP999 missed a deadline.") is None

    def test_matches_unique_first_name_alone(self):
        assert find_employee_id_in_text("Jordan has missed several deadlines") == "EMP001"

    def test_full_name_still_matches(self):
        assert find_employee_id_in_text("Jordan Alvarez has missed several deadlines") == "EMP001"

    def test_employee_id_still_matches(self):
        assert find_employee_id_in_text("EMP001 has missed several deadlines") == "EMP001"

    def test_no_employee_named_still_returns_none(self):
        assert find_employee_id_in_text("My engineer has missed several deadlines") is None

    def test_ambiguous_first_name_returns_none(self, monkeypatch):
        from app.schemas.company_data import Employee

        duplicate_first_name_roster = [
            Employee(
                employee_id="EMP001",
                name="Jordan Alvarez",
                role="Software Engineer",
                team="Platform",
                manager="Sam Rivera",
                start_date="2024-02-12",
            ),
            Employee(
                employee_id="EMP004",
                name="Jordan Smith",
                role="Software Engineer",
                team="Platform",
                manager="Sam Rivera",
                start_date="2025-01-01",
            ),
        ]
        monkeypatch.setattr(
            "app.services.company_data_service._employees",
            lambda: duplicate_first_name_roster,
        )
        assert find_employee_id_in_text("Jordan has missed several deadlines") is None
