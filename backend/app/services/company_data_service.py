"""Read-only access to the mock company dataset (backend/data/company/*.json).

Mirrors the role `retrieval_service.py` plays for the knowledge base: this module
only knows how to load and query the data. It has no opinion about prompts, LLM
providers, or tool-calling — that orchestration lives in `analysis_service.py`.

Every JSON file is loaded once and cached in memory (`lru_cache`) since the dataset
is small, local, and read-only for the lifetime of the process.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from app.schemas.company_data import Employee, FeedbackEntry, JiraIssue, OneOnOne, PullRequest
from app.services.errors import EmployeeNotFoundError

COMPANY_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "company"


def _load(filename: str) -> list[dict]:
    path = COMPANY_DATA_DIR / filename
    with path.open() as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _employees() -> list[Employee]:
    return [Employee.model_validate(r) for r in _load("employees.json")]


@lru_cache(maxsize=1)
def _jira_activity() -> list[JiraIssue]:
    return [JiraIssue.model_validate(r) for r in _load("jira_activity.json")]


@lru_cache(maxsize=1)
def _github_activity() -> list[PullRequest]:
    return [PullRequest.model_validate(r) for r in _load("github_activity.json")]


@lru_cache(maxsize=1)
def _one_on_ones() -> list[OneOnOne]:
    return [OneOnOne.model_validate(r) for r in _load("one_on_ones.json")]


@lru_cache(maxsize=1)
def _feedback_history() -> list[FeedbackEntry]:
    return [FeedbackEntry.model_validate(r) for r in _load("feedback_history.json")]


def _require_known_employee(employee_id: str) -> None:
    known_ids = {e.employee_id for e in _employees()}
    if employee_id not in known_ids:
        raise EmployeeNotFoundError(f"No employee found with id '{employee_id}'.")


def get_employee(employee_id: str) -> Employee:
    """Raises EmployeeNotFoundError if `employee_id` isn't in the dataset."""
    for employee in _employees():
        if employee.employee_id == employee_id:
            return employee
    raise EmployeeNotFoundError(f"No employee found with id '{employee_id}'.")


def get_jira_activity(employee_id: str) -> list[JiraIssue]:
    """Raises EmployeeNotFoundError if `employee_id` isn't in the dataset. Returns an empty
    list (not an error) if the employee exists but has no Jira issues on record."""
    _require_known_employee(employee_id)
    return [r for r in _jira_activity() if r.employee_id == employee_id]


def get_github_activity(employee_id: str) -> list[PullRequest]:
    """Raises EmployeeNotFoundError if `employee_id` isn't in the dataset. Returns an empty
    list (not an error) if the employee exists but has no PRs on record."""
    _require_known_employee(employee_id)
    return [r for r in _github_activity() if r.employee_id == employee_id]


def get_one_on_ones(employee_id: str) -> list[OneOnOne]:
    """Raises EmployeeNotFoundError if `employee_id` isn't in the dataset. Returns an empty
    list (not an error) if the employee exists but has no one-on-one notes on record."""
    _require_known_employee(employee_id)
    return [r for r in _one_on_ones() if r.employee_id == employee_id]


def get_feedback_history(employee_id: str) -> list[FeedbackEntry]:
    """Raises EmployeeNotFoundError if `employee_id` isn't in the dataset. Returns an empty
    list (not an error) if the employee exists but has no feedback on record."""
    _require_known_employee(employee_id)
    return [r for r in _feedback_history() if r.employee_id == employee_id]


def find_employee_id_in_text(text: str) -> str | None:
    """Looks for a known employee (by id, full name, or unique first name, case-insensitive)
    directly mentioned in free text, e.g. a manager's situation description.

    This is the ONLY way an employee gets identified for company-data lookups — the agent
    itself never supplies or chooses an employee_id. That keeps lookups scoped to someone
    the manager actually named, and prevents browsing across the roster.

    Returns None if no known employee is mentioned (including a mentioned id/name that
    doesn't match anyone — this deliberately does not distinguish "no employee named" from
    "an unrecognized employee named" since both should skip company-data lookups), and also
    if a first name mentioned is shared by more than one employee — guessing which one would
    risk attributing evidence to the wrong person, so ambiguous first names are left as None.
    """
    lowered = text.lower()
    for employee in _employees():
        if employee.employee_id.lower() in lowered or employee.name.lower() in lowered:
            return employee.employee_id

    first_name_matches = {
        employee.employee_id
        for employee in _employees()
        if re.search(rf"\b{re.escape(employee.name.split()[0].lower())}\b", lowered)
    }
    if len(first_name_matches) == 1:
        return next(iter(first_name_matches))
    return None
