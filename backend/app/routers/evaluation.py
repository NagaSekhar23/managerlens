from fastapi import APIRouter, HTTPException

from evaluation.report import EvaluationReport, load_latest_report

router = APIRouter(prefix="/api", tags=["evaluation"])


@router.get("/evaluation", response_model=EvaluationReport)
def get_latest_evaluation() -> EvaluationReport:
    report = load_latest_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found yet. Run `python -m evaluation.runner` first.",
        )
    return report
