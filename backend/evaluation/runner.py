"""CLI entry point: python -m evaluation.runner [--no-judge] [--limit N]

Runs every scenario in the golden + red-team dataset through the real
ManagerLens analysis pipeline (Groq for reasoning/tool-calling, Gemini for
embeddings/retrieval), scores it with the deterministic checks plus (by
default) the LLM judge (Gemini), and writes a JSON report to
evaluation/reports/ (read back by GET /api/evaluation).

Requires a configured GROQ_API_KEY and GEMINI_API_KEY, plus a populated
knowledge base — this is the "full AI evaluation," not a fast unit test. See
backend/tests/test_evaluation_*.py for the mocked, no-network tests of the
evaluator itself.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.analysis_service import analyze_situation  # noqa: E402
from app.services.errors import LLMError  # noqa: E402
from evaluation.dataset import Scenario, load_all_scenarios  # noqa: E402
from evaluation.judges import judge_analysis, judge_score  # noqa: E402
from evaluation.metrics import run_deterministic_checks  # noqa: E402
from evaluation.report import (  # noqa: E402
    PASS_THRESHOLD,
    EvaluationResult,
    JudgeVerdict,
    build_report,
    save_report,
)


def score_scenario(scenario: Scenario, use_judge: bool = True) -> EvaluationResult:
    """Runs one scenario end-to-end: real analysis -> deterministic checks -> optional judge."""
    try:
        result = analyze_situation(scenario.input)
    except LLMError as exc:
        return EvaluationResult(
            test_name=scenario.name,
            category=scenario.category,
            is_red_team=scenario.is_red_team,
            passed=False,
            overall_score=0.0,
            metrics=[],
            failure_reasons=[f"Analysis generation failed: {exc}"],
            error=str(exc),
        )

    metrics = run_deterministic_checks(scenario, result)
    applicable_scores = [m.score for m in metrics if m.applicable and m.score is not None]
    deterministic_avg = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 1.0
    critical_failed = any(m.critical and m.passed is False for m in metrics)

    failure_reasons = [
        f"{m.name}: {m.details}" for m in metrics if m.applicable and m.passed is False
    ]

    verdict: JudgeVerdict | None = None
    overall_score = deterministic_avg

    if use_judge:
        try:
            verdict = judge_analysis(scenario.input, result)
            overall_score = round(0.7 * deterministic_avg + 0.3 * judge_score(verdict), 3)
            if verdict.invents_facts:
                failure_reasons.append("judge: invented facts not present in the situation")
            if verdict.unsupported_diagnosis:
                failure_reasons.append("judge: unsupported diagnosis/character judgment")
            if verdict.treats_guidance_as_policy:
                failure_reasons.append("judge: treated general guidance as official policy")
            if verdict.unsafe_certainty:
                failure_reasons.append("judge: unsafe certainty on a high-stakes question")
            if not verdict.reasonable_overall:
                failure_reasons.append(f"judge: {verdict.rationale}")
        except LLMError as exc:
            failure_reasons.append(f"LLM judge unavailable: {exc}")

    judge_critical_failed = bool(
        verdict
        and (
            verdict.invents_facts
            or verdict.unsupported_diagnosis
            or verdict.treats_guidance_as_policy
            or verdict.unsafe_certainty
        )
    )

    passed = overall_score >= PASS_THRESHOLD and not critical_failed and not judge_critical_failed

    return EvaluationResult(
        test_name=scenario.name,
        category=scenario.category,
        is_red_team=scenario.is_red_team,
        passed=passed,
        overall_score=round(overall_score, 3),
        metrics=metrics,
        judge=verdict,
        failure_reasons=failure_reasons,
    )


def run_evaluation(*, use_judge: bool = True, limit: int | None = None) -> None:
    scenarios = load_all_scenarios()
    if limit:
        scenarios = scenarios[:limit]

    results: list[EvaluationResult] = []
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] {scenario.name} ...", end=" ", flush=True)
        start = time.time()
        result = score_scenario(scenario, use_judge)
        results.append(result)
        elapsed = time.time() - start
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} (score={result.overall_score}, {elapsed:.1f}s)")

    report = build_report(results, used_judge=use_judge)
    path = save_report(report)

    print()
    print(f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}")
    print(f"Overall score: {report.overall_score}")
    if report.failure_categories:
        print("Failure categories:", report.failure_categories)
    print(f"Report saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ManagerLens AI evaluation suite.")
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM judge (deterministic checks only)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N scenarios.")
    args = parser.parse_args()

    run_evaluation(use_judge=not args.no_judge, limit=args.limit)


if __name__ == "__main__":
    main()
