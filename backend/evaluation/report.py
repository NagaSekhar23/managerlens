"""Pydantic models for evaluation output, plus JSON persistence for the
latest report (read by both the CLI runner and GET /api/evaluation)."""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
LATEST_REPORT_PATH = REPORTS_DIR / "latest.json"

PASS_THRESHOLD = 0.7


class MetricScore(BaseModel):
    name: str
    applicable: bool
    passed: bool | None = None
    score: float | None = None
    critical: bool = False
    details: str = ""


class JudgeVerdict(BaseModel):
    invents_facts: bool
    unsupported_diagnosis: bool
    treats_guidance_as_policy: bool
    unsafe_certainty: bool
    reasonable_overall: bool
    rationale: str


class EvaluationResult(BaseModel):
    test_name: str
    category: str
    is_red_team: bool
    passed: bool
    overall_score: float
    metrics: list[MetricScore]
    judge: JudgeVerdict | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    error: str | None = None


class EvaluationReport(BaseModel):
    generated_at: datetime
    total: int
    passed: int
    failed: int
    overall_score: float
    metric_averages: dict[str, float]
    failure_categories: dict[str, int]
    used_judge: bool
    results: list[EvaluationResult]


def build_report(results: list[EvaluationResult], *, used_judge: bool) -> EvaluationReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    metric_totals: dict[str, list[float]] = {}
    for r in results:
        for m in r.metrics:
            if m.applicable and m.score is not None:
                metric_totals.setdefault(m.name, []).append(m.score)
    metric_averages = {
        name: round(sum(scores) / len(scores), 3) for name, scores in metric_totals.items()
    }

    failure_categories: dict[str, int] = {}
    for r in results:
        if not r.passed:
            failure_categories[r.category] = failure_categories.get(r.category, 0) + 1

    overall_score = round(sum(r.overall_score for r in results) / total, 3) if total else 0.0

    return EvaluationReport(
        generated_at=datetime.now(timezone.utc),
        total=total,
        passed=passed,
        failed=total - passed,
        overall_score=overall_score,
        metric_averages=metric_averages,
        failure_categories=failure_categories,
        used_judge=used_judge,
        results=results,
    )


def save_report(report: EvaluationReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    dated_path = REPORTS_DIR / f"report-{timestamp}.json"

    payload = report.model_dump(mode="json")
    dated_path.write_text(json.dumps(payload, indent=2))
    LATEST_REPORT_PATH.write_text(json.dumps(payload, indent=2))

    return dated_path


def load_latest_report() -> EvaluationReport | None:
    if not LATEST_REPORT_PATH.exists():
        return None
    data = json.loads(LATEST_REPORT_PATH.read_text())
    return EvaluationReport.model_validate(data)
