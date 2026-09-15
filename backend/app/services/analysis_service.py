"""Builds the ManagerLens prompt, retrieves supporting evidence, and turns a
situation into a structured, grounded analysis.

Manager situation -> retrieve relevant knowledge -> build prompt with
evidence -> Gemini structured output -> AnalysisResult (with knowledge_used
attached).
"""

import logging

from app.schemas.analysis import AnalysisResult, GeminiAnalysisPayload, KnowledgeSource
from app.services.errors import KnowledgeRetrievalError
from app.services.gemini_client import generate_structured
from app.services.retrieval_service import KnowledgeMatch, retrieve_relevant_knowledge

logger = logging.getLogger(__name__)

EXCERPT_LENGTH = 280

SYSTEM_INSTRUCTION = """You are ManagerLens, an assistant that helps managers think clearly \
about workplace situations before they act. You are not a therapist, HR system, or judge of \
the employee's character.

Hard rules, in priority order:
1. Never diagnose the employee (no mental-health, personality, or motive labels — e.g. do not \
   say someone "is disengaged", "is depressed", "doesn't care"; instead describe only what was \
   observed and label motive-based interpretations as assumptions to avoid).
2. Never invent facts. Only put something in `observed_facts` if the manager's text directly \
   states it. If you are inferring or guessing, it belongs in `assumptions_to_avoid`, \
   `missing_context`, or is simply left out.
3. Strictly separate three categories:
   - observed_facts: literally stated by the manager (events, counts, dates, direct quotes, \
     directly reported behavior).
   - assumptions_to_avoid: plausible-sounding interpretations, motives, causes, or diagnoses \
     that the manager (or a reader) might jump to, which are NOT established by the facts and \
     should explicitly be held loosely, not acted on as if true.
   - missing_context / clarifying_questions: what is unknown and would change the picture.
4. Every recommended_action and conversation_plan item must be practical and grounded only in \
   the observed facts, framed conditionally ("if X turns out to be true...") wherever it \
   depends on something currently unknown or assumed.
5. Acknowledge uncertainty honestly in `confidence` and `reasoning_basis`. If little was stated, \
   confidence must be low and reasoning_basis must say so plainly.
6. Return ONLY the structured fields defined by the response schema — no extra commentary, no \
   markdown, no text outside the schema.

You may also be given "Retrieved evidence": excerpts from a general management knowledge base,
each with a relevance score. Rules for using this evidence:
7. Retrieved evidence is general management guidance, written for no specific person or company.
   It is supporting evidence for how to think about situations like this — never treat it as a
   fact about the employee, the manager, or this specific team.
8. Do not invent policies, statistics, studies, or citations beyond what's given in the evidence.
   If the evidence doesn't say something, don't attribute it to the evidence.
9. If the evidence is weak, generic, off-topic, or absent, say so plainly in `reasoning_basis` and
   lower `confidence` accordingly — do not stretch thin evidence to sound more authoritative.
10. Where a recommendation or conversation_plan item draws on retrieved evidence, phrase it so the
    reader can tell that it's general guidance being applied here, not a fact about their
    situation (e.g. "general guidance suggests separating skill gaps from motivation gaps —
    consider which this looks more like" rather than stating it as settled).
"""


def _build_evidence_block(matches: list[KnowledgeMatch]) -> str:
    if not matches:
        return (
            "No relevant guidance was found in the knowledge base for this situation. "
            "Proceed using only the situation itself, and reflect this absence of evidence "
            "in your confidence and reasoning_basis."
        )

    lines = ["Retrieved evidence (general management guidance, not facts about this employee):"]
    for i, match in enumerate(matches, start=1):
        lines.append(f'{i}. [{match.title}] (relevance {match.score:.2f}): "{match.content}"')
    return "\n\n".join(lines)


def _build_prompt(situation: str, matches: list[KnowledgeMatch]) -> str:
    return (
        "A manager submitted the following workplace situation. Analyze it according to your "
        "system instructions and return the structured analysis.\n\n"
        f"Situation:\n{situation}\n\n"
        f"{_build_evidence_block(matches)}"
    )


def _clean_excerpt_text(content: str) -> str:
    """Strips markdown headers/blockquote markers and collapses whitespace so the excerpt
    reads as plain prose in the UI."""
    lines = []
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").lstrip(">").strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def _to_knowledge_source(match: KnowledgeMatch) -> KnowledgeSource:
    excerpt = _clean_excerpt_text(match.content)
    if len(excerpt) > EXCERPT_LENGTH:
        excerpt = excerpt[:EXCERPT_LENGTH].rsplit(" ", 1)[0] + "…"

    return KnowledgeSource(
        title=match.title,
        excerpt=excerpt,
        relevance_score=round(match.score, 3),
    )


def analyze_situation(situation: str) -> AnalysisResult:
    """Run the ManagerLens analysis for a single manager-submitted situation.

    Retrieval failures (knowledge-base outage) degrade gracefully: the analysis
    proceeds without evidence rather than failing the whole request, since RAG
    evidence is supporting input, not a hard dependency. Gemini failures
    (MissingAPIKeyError, LLMRequestError, InvalidLLMOutputError) are NOT
    caught here — the router translates those into HTTP responses.
    """
    try:
        matches = retrieve_relevant_knowledge(situation)
    except KnowledgeRetrievalError as exc:
        logger.error("Knowledge retrieval failed, proceeding without evidence: %s", exc)
        matches = []

    payload = generate_structured(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=_build_prompt(situation, matches),
        response_model=GeminiAnalysisPayload,
    )

    return AnalysisResult(
        **payload.model_dump(),
        knowledge_used=[_to_knowledge_source(m) for m in matches],
    )
