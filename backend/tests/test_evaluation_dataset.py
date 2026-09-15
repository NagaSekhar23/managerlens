from evaluation.dataset import RED_TEAM_SCENARIOS, SCENARIOS, load_all_scenarios

REQUIRED_CATEGORIES = {
    "performance_problem",
    "missed_deadlines",
    "conflict",
    "difficult_feedback",
    "one_on_one",
    "unclear_situation",
    "insufficient_context",
    "disengagement_claim",
    "manager_assumption",
    "diagnosis_request",
    "termination_request",
    "off_topic",
    "weak_retrieval",
    "strong_retrieval",
    "ambiguous_situation",
}


def test_at_least_25_main_scenarios():
    assert len(SCENARIOS) >= 25


def test_all_required_categories_covered():
    categories = {s.category for s in load_all_scenarios()}
    missing = REQUIRED_CATEGORIES - categories
    assert not missing, f"Missing categories: {missing}"


def test_red_team_dataset_is_nonempty_and_flagged():
    assert len(RED_TEAM_SCENARIOS) >= 4
    assert all(s.is_red_team for s in RED_TEAM_SCENARIOS)


def test_scenario_names_are_unique():
    names = [s.name for s in load_all_scenarios()]
    assert len(names) == len(set(names))


def test_every_scenario_has_a_nonempty_input_and_at_least_one_expectation():
    for scenario in load_all_scenarios():
        assert scenario.input.strip() != ""
        # A scenario with no expectations set would trivially "pass" everything.
        assert scenario.expected.model_dump(exclude_defaults=True) != {}
