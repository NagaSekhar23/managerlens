"""The ManagerLens evaluation dataset: a golden set of manager scenarios with
expected AI behavior, plus a red-team set specifically designed to provoke
unsafe or unreliable outputs (diagnosis, invented facts, fabricated policy,
termination directives, overconfidence).
"""

from evaluation.dataset.red_team import RED_TEAM_SCENARIOS
from evaluation.dataset.scenarios import SCENARIOS
from evaluation.dataset.schema import Scenario, ScenarioExpectation


def load_all_scenarios() -> list[Scenario]:
    return [*SCENARIOS, *RED_TEAM_SCENARIOS]


__all__ = ["Scenario", "ScenarioExpectation", "SCENARIOS", "RED_TEAM_SCENARIOS", "load_all_scenarios"]
