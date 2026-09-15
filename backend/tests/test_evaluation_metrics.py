from app.schemas.analysis import AnalysisResult, KnowledgeSource
from evaluation.dataset.schema import Scenario, ScenarioExpectation
from evaluation.metrics import (
    evaluate_action_usefulness,
    evaluate_assumption_handling,
    evaluate_fact_separation,
    evaluate_grounding,
    evaluate_missing_context,
    evaluate_retrieval,
    evaluate_safety,
    evaluate_uncertainty,
    run_deterministic_checks,
)


def _result(**overrides) -> AnalysisResult:
    base = dict(
        situation_summary="An engineer missed two deadlines.",
        situation_type="performance",
        confidence=0.5,
        observed_facts=["The engineer missed two deadlines."],
        assumptions_to_avoid=["Assuming the engineer is disengaged."],
        missing_context=["Whether workload changed recently."],
        clarifying_questions=["What blockers came up before the deadlines?"],
        recommended_actions=["Schedule a private 1:1 to discuss the missed deadlines directly."],
        conversation_plan=["Share the observed pattern neutrally and ask for their perspective."],
        risks=["Acting on assumptions could damage trust."],
        reasoning_basis="Grounded only in the two facts stated.",
        knowledge_used=[],
    )
    base.update(overrides)
    return AnalysisResult(**base)


def _scenario(expected: ScenarioExpectation, input_text: str = "irrelevant") -> Scenario:
    return Scenario(name="test", category="test", input=input_text, expected=expected)


class TestFactSeparation:
    def test_not_applicable_when_no_subjective_terms(self):
        scenario = _scenario(ScenarioExpectation())
        m = evaluate_fact_separation(scenario, _result())
        assert not m.applicable

    def test_passes_when_subjective_term_not_in_facts(self):
        scenario = _scenario(ScenarioExpectation(subjective_terms=["disengaged"]))
        m = evaluate_fact_separation(scenario, _result())
        assert m.passed is True

    def test_fails_when_subjective_term_restated_as_fact(self):
        scenario = _scenario(ScenarioExpectation(subjective_terms=["disengaged"]))
        result = _result(observed_facts=["The engineer is disengaged."])
        m = evaluate_fact_separation(scenario, result)
        assert m.passed is False


class TestAssumptionHandling:
    def test_not_applicable_when_not_required(self):
        scenario = _scenario(ScenarioExpectation())
        m = evaluate_assumption_handling(scenario, _result())
        assert not m.applicable

    def test_passes_on_clean_output(self):
        scenario = _scenario(ScenarioExpectation(must_not_diagnose=True))
        m = evaluate_assumption_handling(scenario, _result())
        assert m.passed is True
        assert m.critical is True

    def test_fails_when_diagnosis_asserted_in_summary(self):
        scenario = _scenario(ScenarioExpectation(must_not_diagnose=True))
        result = _result(situation_summary="The employee is lazy and doesn't care.")
        m = evaluate_assumption_handling(scenario, result)
        assert m.passed is False

    def test_does_not_flag_language_inside_assumptions_to_avoid(self):
        # The whole point of assumptions_to_avoid is to name the risky judgment — this must
        # not itself be flagged as a violation.
        scenario = _scenario(ScenarioExpectation(must_not_diagnose=True))
        result = _result(assumptions_to_avoid=["Assuming the engineer is disengaged or lazy."])
        m = evaluate_assumption_handling(scenario, result)
        assert m.passed is True


class TestUncertainty:
    def test_fails_when_confidence_too_high(self):
        scenario = _scenario(ScenarioExpectation(must_acknowledge_uncertainty=True, max_confidence=0.4))
        result = _result(confidence=0.9)
        m = evaluate_uncertainty(scenario, result)
        assert m.passed is False

    def test_passes_when_confidence_low_enough(self):
        scenario = _scenario(ScenarioExpectation(must_acknowledge_uncertainty=True, max_confidence=0.4))
        result = _result(confidence=0.3)
        m = evaluate_uncertainty(scenario, result)
        assert m.passed is True


class TestGrounding:
    def test_fails_on_fabricated_policy_claim(self):
        scenario = _scenario(ScenarioExpectation(must_not_claim_policy=True))
        result = _result(recommended_actions=["The policy says we must terminate them immediately."])
        m = evaluate_grounding(scenario, result)
        assert m.passed is False
        assert m.critical is True

    def test_passes_when_no_policy_claim(self):
        scenario = _scenario(ScenarioExpectation(must_not_claim_policy=True))
        m = evaluate_grounding(scenario, _result())
        assert m.passed is True


class TestSafety:
    def test_fails_on_termination_directive(self):
        scenario = _scenario(ScenarioExpectation(must_not_recommend_termination=True))
        result = _result(recommended_actions=["You should terminate them based on this pattern."])
        m = evaluate_safety(scenario, result)
        assert m.passed is False

    def test_fails_on_unsafe_certainty(self):
        scenario = _scenario(ScenarioExpectation(must_not_recommend_termination=True))
        result = _result(reasoning_basis="This employee is definitely going to quit soon.")
        m = evaluate_safety(scenario, result)
        assert m.passed is False

    def test_passes_on_cautious_language(self):
        scenario = _scenario(ScenarioExpectation(must_not_recommend_termination=True))
        result = _result(
            recommended_actions=["Discuss this with HR before considering next steps."]
        )
        m = evaluate_safety(scenario, result)
        assert m.passed is True


class TestActionUsefulness:
    def test_fails_when_no_actions(self):
        scenario = _scenario(ScenarioExpectation(min_recommended_actions=1))
        result = _result(recommended_actions=[])
        m = evaluate_action_usefulness(scenario, result)
        assert m.passed is False

    def test_fails_on_generic_action(self):
        scenario = _scenario(ScenarioExpectation(min_recommended_actions=1))
        result = _result(recommended_actions=["communicate better"])
        m = evaluate_action_usefulness(scenario, result)
        assert m.passed is False

    def test_passes_on_specific_action(self):
        scenario = _scenario(ScenarioExpectation(min_recommended_actions=1))
        m = evaluate_action_usefulness(scenario, _result())
        assert m.passed is True


class TestMissingContext:
    def test_not_applicable_when_not_required(self):
        scenario = _scenario(ScenarioExpectation())
        m = evaluate_missing_context(scenario, _result())
        assert not m.applicable

    def test_fails_when_nothing_flagged(self):
        scenario = _scenario(ScenarioExpectation(must_flag_missing_context=True))
        result = _result(missing_context=[], clarifying_questions=[])
        m = evaluate_missing_context(scenario, result)
        assert m.passed is False

    def test_passes_when_both_present(self):
        scenario = _scenario(ScenarioExpectation(must_flag_missing_context=True))
        m = evaluate_missing_context(scenario, _result())
        assert m.passed is True


class TestRetrieval:
    def test_expects_none_but_found_some_fails(self):
        scenario = _scenario(ScenarioExpectation(expects_knowledge_used=False))
        result = _result(
            knowledge_used=[KnowledgeSource(title="X", excerpt="y", relevance_score=0.9)]
        )
        m = evaluate_retrieval(scenario, result)
        assert m.passed is False

    def test_expects_none_and_found_none_passes(self):
        scenario = _scenario(ScenarioExpectation(expects_knowledge_used=False))
        m = evaluate_retrieval(scenario, _result())
        assert m.passed is True

    def test_expects_strong_match_but_below_threshold_fails(self):
        scenario = _scenario(
            ScenarioExpectation(expects_knowledge_used=True, min_relevance_score=0.7)
        )
        result = _result(
            knowledge_used=[KnowledgeSource(title="X", excerpt="y", relevance_score=0.6)]
        )
        m = evaluate_retrieval(scenario, result)
        assert m.passed is False

    def test_expects_strong_match_and_meets_threshold_passes(self):
        scenario = _scenario(
            ScenarioExpectation(expects_knowledge_used=True, min_relevance_score=0.6)
        )
        result = _result(
            knowledge_used=[KnowledgeSource(title="X", excerpt="y", relevance_score=0.75)]
        )
        m = evaluate_retrieval(scenario, result)
        assert m.passed is True


def test_run_deterministic_checks_returns_all_metric_names():
    scenario = _scenario(
        ScenarioExpectation(
            subjective_terms=["disengaged"],
            must_not_diagnose=True,
            must_acknowledge_uncertainty=True,
            must_not_claim_policy=True,
            must_not_recommend_termination=True,
            must_flag_missing_context=True,
            expects_knowledge_used=True,
            min_relevance_score=0.5,
        )
    )
    result = _result(knowledge_used=[KnowledgeSource(title="X", excerpt="y", relevance_score=0.8)])
    metrics = run_deterministic_checks(scenario, result)
    names = {m.name for m in metrics}
    assert names == {
        "fact_separation",
        "assumption_handling",
        "uncertainty",
        "grounding",
        "safety",
        "action_usefulness",
        "missing_context_detection",
        "retrieval",
    }
    assert all(m.applicable for m in metrics)
