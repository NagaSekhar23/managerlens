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
        # MissingAPIKeyError/LLMRequestError/InvalidLLMOutputError are provider-agnostic (see
        # app/services/errors.py) — analyze_situation() can raise any of them from either the
        # Groq calls (reasoning, tool calling) or the Gemini calls (embeddings/RAG), so the log
        # message here stays provider-neutral. The exception's own message (from whichever
        # client raised it) already names the specific provider.
        logger.error("LLM provider API key missing: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The analysis service is not configured. Please contact the administrator.",
        ) from exc
    except LLMRequestError as exc:
        logger.error("LLM provider request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI service is temporarily unavailable. Please try again in a moment.",
        ) from exc
    except InvalidLLMOutputError as exc:
        logger.error("LLM provider returned malformed output: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI returned an unexpected response. Please try again.",
        ) from exc
