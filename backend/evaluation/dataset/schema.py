"""Defines what a test case in the evaluation dataset looks like.

`ScenarioExpectation` is deliberately a set of optional flags/thresholds
rather than one rigid shape: different scenario categories care about
different things (an off-topic question doesn't need a `min_relevance_score`,
a strong-retrieval scenario doesn't need `must_not_recommend_termination`).
A field left as None/empty means "don't check this for this scenario" —
metrics.py treats it as not applicable rather than a failure.
"""

from pydantic import BaseModel, Field


class ScenarioExpectation(BaseModel):
    # Fact / assumption separation
    subjective_terms: list[str] = Field(
        default_factory=list,
        description="Judgmental words/phrases present in the input that must NOT be restated "
        "as observed_facts — they belong in assumptions_to_avoid instead, if anywhere.",
    )
    min_observed_facts: int | None = None

    # Diagnosis / safety
    must_not_diagnose: bool = False
    must_not_recommend_termination: bool = False
    must_not_claim_policy: bool = False

    # Uncertainty
    must_acknowledge_uncertainty: bool = False
    max_confidence: float = 0.5

    # Missing context
    must_flag_missing_context: bool = False

    # Action usefulness
    min_recommended_actions: int = 1

    # Retrieval
    expects_knowledge_used: bool | None = None
    min_relevance_score: float | None = None


class Scenario(BaseModel):
    name: str
    category: str
    input: str
    expected: ScenarioExpectation
    is_red_team: bool = False
    notes: str = ""
