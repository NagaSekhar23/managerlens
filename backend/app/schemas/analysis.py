"""Pydantic schemas for POST /api/analyze.

`GeminiAnalysisPayload` is exactly what the reasoning provider's structured-output
call (currently Groq — see `groq_client.generate_structured`) is asked to produce.
The class name predates the Groq migration and was kept as-is to avoid a broad
rename across schemas/tests/evaluation. `AnalysisResult` extends it with
`knowledge_used` and `company_data_used`, which are populated by our own
retrieval/tool-calling code — never by the model itself — because letting the
model fill in its own "sources" would defeat the point of grounding it in real,
retrieved evidence.
"""

from typing import Literal

from pydantic import BaseModel, Field

MAX_SITUATION_LENGTH = 4000

SituationType = Literal[
    "performance",
    "engagement",
    "conduct",
    "communication",
    "workload",
    "interpersonal_conflict",
    "career_development",
    "other",
]


class AnalyzeRequest(BaseModel):
    situation: str = Field(..., min_length=1, max_length=MAX_SITUATION_LENGTH)


class GeminiAnalysisPayload(BaseModel):
    situation_summary: str = Field(
        ..., description="A short, neutral restatement of what the manager described."
    )
    situation_type: SituationType = Field(
        ..., description="The single category that best fits the situation."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well-supported this analysis is, given how much is known vs. unknown.",
    )
    observed_facts: list[str] = Field(
        default_factory=list,
        description="Only statements the manager explicitly reported (events, counts, dates, direct quotes).",
    )
    assumptions_to_avoid: list[str] = Field(
        default_factory=list,
        description="Plausible-sounding interpretations, motives, or diagnoses that are NOT supported by the facts alone and should not be treated as true.",
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Information that is absent and would materially change the interpretation or the advice.",
    )
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="Concrete questions the manager should answer, or ask the employee, to fill the missing context.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Practical next steps grounded in the observed facts, framed conditionally where they depend on an assumption or open question.",
    )
    conversation_plan: list[str] = Field(
        default_factory=list,
        description="An ordered set of talking points for a 1:1 conversation with the employee.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risks of acting on this situation too early, or of not acting at all.",
    )
    reasoning_basis: str = Field(
        ...,
        description="One or two sentences on what this analysis is and is not grounded in (e.g. facts only, or conditional on assumptions).",
    )


class KnowledgeSource(BaseModel):
    """One piece of retrieved evidence shown to the user — no internal IDs, file paths, or
    database details, just what a human needs to judge the evidence themselves."""

    title: str = Field(..., description="The knowledge document this excerpt came from.")
    excerpt: str = Field(..., description="A short excerpt of the retrieved guidance text.")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity between the situation and this chunk."
    )


class CompanyDataSource(BaseModel):
    """One piece of observed company evidence used as input — a compact, human-readable
    summary of what a tool actually returned, never the raw record dump."""

    tool: str = Field(..., description="Which company-data tool produced this evidence, e.g. get_jira_activity.")
    employee_id: str = Field(..., description="The employee this evidence is about.")
    employee_name: str = Field(..., description="Display name of the employee this evidence is about.")
    summary: str = Field(
        ...,
        description="Concise, human-readable summary of what the tool actually returned (e.g. specific issue ids, dates, or note excerpts) — never invented.",
    )


class AnalysisResult(GeminiAnalysisPayload):
    knowledge_used: list[KnowledgeSource] = Field(
        default_factory=list,
        description="Retrieved general management guidance used as supporting evidence, if any was relevant.",
    )
    company_data_used: list[CompanyDataSource] = Field(
        default_factory=list,
        description="Observed company-data evidence (Jira, GitHub, 1:1s, feedback) retrieved for the employee identified in the situation, if any.",
    )
