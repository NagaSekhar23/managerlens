"""Typed models for the mock company dataset (backend/data/company/*.json).

These mirror the JSON shape exactly — they exist so the company-data tools
return typed objects instead of raw dicts, the same way the rest of the app
avoids passing loosely-typed data between layers.
"""

from pydantic import BaseModel


class Employee(BaseModel):
    employee_id: str
    name: str
    role: str
    team: str
    manager: str
    start_date: str


class JiraIssue(BaseModel):
    employee_id: str
    issue_id: str
    title: str
    status: str
    due_date: str
    completed_date: str | None
    blocked_by: str | None
    project: str


class PullRequest(BaseModel):
    employee_id: str
    repository: str
    pull_request: int
    title: str
    opened_at: str
    merged_at: str | None
    review_count: int


class OneOnOne(BaseModel):
    employee_id: str
    date: str
    topics: list[str]
    concerns: list[str]
    manager_notes: str


class FeedbackEntry(BaseModel):
    employee_id: str
    date: str
    feedback_type: str
    summary: str
