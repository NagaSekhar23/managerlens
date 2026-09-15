import logging

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import AnalysisResult, AnalyzeRequest
from app.services.analysis_service import analyze_situation
from app.services.errors import InvalidLLMOutputError, LLMRequestError, MissingAPIKeyError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalyzeRequest) -> AnalysisResult:
    situation = request.situation.strip()
    if not situation:
        raise HTTPException(status_code=400, detail="Situation must not be empty.")

    try:
        return analyze_situation(situation)
    except MissingAPIKeyError as exc:
        logger.error("Gemini API key missing: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The analysis service is not configured. Please contact the administrator.",
        ) from exc
    except LLMRequestError as exc:
        logger.error("Gemini request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI service is temporarily unavailable. Please try again in a moment.",
        ) from exc
    except InvalidLLMOutputError as exc:
        logger.error("Gemini returned malformed output: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI returned an unexpected response. Please try again.",
        ) from exc
