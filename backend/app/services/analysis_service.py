"""Builds the ManagerLens prompt, retrieves supporting evidence, and turns a
situation into a structured, grounded analysis.

Manager situation -> retrieve relevant knowledge (Gemini embeddings, unchanged)
-> if an employee is named, run a bounded company-data tool loop for that
employee only (Groq) -> build prompt with both kinds of evidence -> Groq
structured output -> AnalysisResult (with knowledge_used and
company_data_used attached).
"""

import json
import logging

from app.config import settings
from app.schemas.analysis import (
    AnalysisResult,
    CompanyDataSource,
    GeminiAnalysisPayload,
    KnowledgeSource,
)
from app.services.company_data_service import (
    find_employee_id_in_text,
    get_employee,
    get_feedback_history,
    get_github_activity,
    get_jira_activity,
    get_one_on_ones,
)
from app.services.errors import EmployeeNotFoundError, KnowledgeRetrievalError, LLMRequestError
from app.services.gemini_client import ToolCallRecord, ToolDeclaration
from app.services.groq_client import generate_structured, run_tool_loop
from app.services.retrieval_service import KnowledgeMatch, retrieve_relevant_knowledge

logger = logging.getLogger(__name__)

EXCERPT_LENGTH = 280

# Tools declared with no parameters — the employee they operate on is bound server-side
# (via closure, see `_build_company_tools`) to whichever employee was identified in the
# manager's own text. The model can choose whether/how many times to call each tool, but
# can never choose *which* employee, and never browse across the roster.
_NO_PARAMETERS_SCHEMA = {"type": "object", "properties": {}}

TOOL_SYSTEM_INSTRUCTION = """You are gathering observed company evidence about ONE specific \
employee, already identified from a manager's situation description, before ManagerLens \
produces its analysis.

You have five read-only tools: get_employee, get_jira_activity, get_github_activity, \
get_one_on_ones, get_feedback_history. None of them take arguments — each one always returns \
data for the single employee already identified, never a different employee.

Call whichever of these tools would help verify or add context to what the manager described. \
You may call more than one, but calling the same tool twice will not return new information. \
Once you have enough evidence (or have determined none of the tools are relevant), stop \
calling tools and reply with a brief plain-text acknowledgement — that final text is discarded; \
only your tool calls matter. Never claim a tool returned something it did not."""

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

You may also be given "Observed company evidence": tool results (Jira, GitHub, one-on-one, and
feedback records) retrieved directly from the company's own systems for the one employee named
in the situation. Rules for using this evidence:
11. Unlike retrieved knowledge, this IS evidence about the specific employee and situation — treat
    each item as an observed_fact (not an assumption), but only exactly what the item states.
    Never extend it to a conclusion the data doesn't itself support.
12. If a lookup returned no records, or no company evidence was retrieved at all, say so plainly
    and treat that absence as missing_context rather than filling the gap with a guess — sparse
    or absent company data is not itself evidence of a problem.
13. Never attribute a piece of observed company evidence to any employee other than the one it
    was retrieved for.
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


def _build_company_tools(employee_id: str) -> list[ToolDeclaration]:
    """The five approved company-data tools, each bound to `employee_id` via closure so
    the model can decide whether/how many times to call them but never which employee
    they run against — that's fixed by our own identification step, not by the model."""
    return [
        ToolDeclaration(
            name="get_employee",
            description="Look up HR directory info (role, team, manager, start date) for the identified employee.",
            parameters_schema=_NO_PARAMETERS_SCHEMA,
            handler=lambda _args: get_employee(employee_id).model_dump(),
        ),
        ToolDeclaration(
            name="get_jira_activity",
            description="Look up Jira issue history (assigned, due, completed, blockers) for the identified employee.",
            parameters_schema=_NO_PARAMETERS_SCHEMA,
            handler=lambda _args: [i.model_dump() for i in get_jira_activity(employee_id)],
        ),
        ToolDeclaration(
            name="get_github_activity",
            description="Look up GitHub pull request history for the identified employee.",
            parameters_schema=_NO_PARAMETERS_SCHEMA,
            handler=lambda _args: [p.model_dump() for p in get_github_activity(employee_id)],
        ),
        ToolDeclaration(
            name="get_one_on_ones",
            description="Look up private 1:1 notes for the identified employee.",
            parameters_schema=_NO_PARAMETERS_SCHEMA,
            handler=lambda _args: [o.model_dump() for o in get_one_on_ones(employee_id)],
        ),
        ToolDeclaration(
            name="get_feedback_history",
            description="Look up feedback given about the identified employee, from managers, peers, or performance reviews.",
            parameters_schema=_NO_PARAMETERS_SCHEMA,
            handler=lambda _args: [f.model_dump() for f in get_feedback_history(employee_id)],
        ),
    ]


def _build_tool_prompt(situation: str, employee_id: str) -> str:
    return (
        "A manager submitted the following workplace situation, which names employee "
        f"{employee_id}. Call whichever of your available tools would help verify or add "
        "context to what's described, then reply with a brief closing note.\n\n"
        f"Situation:\n{situation}"
    )


def _build_company_evidence_block(records: list[ToolCallRecord], employee_id: str | None) -> str:
    if employee_id is None:
        return ""

    if not records:
        return (
            f"No company-data tools were called for employee {employee_id}. Proceed without "
            "company evidence for this employee, and reflect that absence in missing_context."
        )

    lines = [
        f"Observed company evidence for employee {employee_id} "
        "(retrieved directly from company systems, not stated by the manager):"
    ]
    for record in records:
        if record.error:
            lines.append(f"- {record.name}: lookup failed ({record.error}). Assume nothing from this.")
        elif not record.result:
            lines.append(f"- {record.name}: no records found.")
        else:
            lines.append(f"- {record.name}: {json.dumps(record.result)}")
    return "\n".join(lines)


def _build_prompt(
    situation: str,
    matches: list[KnowledgeMatch],
    employee_id: str | None,
    tool_records: list[ToolCallRecord],
) -> str:
    sections = [
        "A manager submitted the following workplace situation. Analyze it according to your "
        "system instructions and return the structured analysis.\n\n"
        f"Situation:\n{situation}",
        _build_evidence_block(matches),
    ]
    company_block = _build_company_evidence_block(tool_records, employee_id)
    if company_block:
        sections.append(company_block)
    return "\n\n".join(sections)


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


COMPANY_SUMMARY_LENGTH = 280


def _summarize_company_record(tool_name: str, result: object) -> str:
    """Turns one tool's raw (already-dict) result into a short, concrete summary — built
    only from fields the tool actually returned, never inferred or invented."""
    if not result:
        return "No records found."

    if tool_name == "get_employee" and isinstance(result, dict):
        return (
            f"{result.get('role')} on the {result.get('team')} team, reports to "
            f"{result.get('manager')} (started {result.get('start_date')})."
        )

    if not isinstance(result, list):
        return "Record found."

    count = len(result)
    if tool_name == "get_jira_activity":
        items = [
            f"{i.get('issue_id')} \"{i.get('title')}\" — {i.get('status')}"
            + (
                f", completed {i['completed_date']}"
                if i.get("completed_date")
                else f", due {i.get('due_date')}"
            )
            for i in result
        ]
    elif tool_name == "get_github_activity":
        items = [
            f"PR #{p.get('pull_request')} \"{p.get('title')}\""
            + (f", merged {p['merged_at']}" if p.get("merged_at") else ", still open")
            for p in result
        ]
    elif tool_name == "get_one_on_ones":
        items = [f"{o.get('date')}: {o.get('manager_notes')}" for o in result]
    elif tool_name == "get_feedback_history":
        items = [f"{f.get('date')} ({f.get('feedback_type')}): {f.get('summary')}" for f in result]
    else:
        items = []

    summary = f"{count} record(s) found. " + "; ".join(items)
    if len(summary) > COMPANY_SUMMARY_LENGTH:
        summary = summary[:COMPANY_SUMMARY_LENGTH].rsplit(" ", 1)[0] + "…"
    return summary


def _to_company_data_source(
    record: ToolCallRecord, employee_id: str, employee_name: str
) -> CompanyDataSource:
    return CompanyDataSource(
        tool=record.name,
        employee_id=employee_id,
        employee_name=employee_name,
        summary=_summarize_company_record(record.name, record.result),
    )


def analyze_situation(situation: str) -> AnalysisResult:
    """Run the ManagerLens analysis for a single manager-submitted situation.

    Retrieval failures (knowledge-base outage) degrade gracefully: the analysis
    proceeds without evidence rather than failing the whole request, since RAG
    evidence is supporting input, not a hard dependency. RAG retrieval/embeddings stay
    on Gemini; reasoning, tool calling, and the final structured answer run on Groq.
    Provider failures (MissingAPIKeyError, LLMRequestError, InvalidLLMOutputError) are
    NOT caught here — the router translates those into HTTP responses.

    Company-data tools are only ever offered to the model if `find_employee_id_in_text`
    recognizes an employee actually named in the manager's own text — the model never
    chooses or browses across employees, and a tool-loop failure degrades gracefully the
    same way a knowledge-retrieval failure does.
    """
    try:
        matches = retrieve_relevant_knowledge(situation)
    except KnowledgeRetrievalError as exc:
        logger.error("Knowledge retrieval failed, proceeding without evidence: %s", exc)
        matches = []

    employee_id = find_employee_id_in_text(situation)
    employee_name = employee_id
    tool_records: list[ToolCallRecord] = []
    if employee_id is not None:
        try:
            employee_name = get_employee(employee_id).name
        except EmployeeNotFoundError:  # pragma: no cover - identification guarantees this exists
            employee_name = employee_id

        try:
            tool_records = run_tool_loop(
                system_instruction=TOOL_SYSTEM_INSTRUCTION,
                prompt=_build_tool_prompt(situation, employee_id),
                tools=_build_company_tools(employee_id),
                max_tool_calls=settings.max_tool_calls_per_analysis,
            )
        except LLMRequestError as exc:
            logger.error("Company-data tool loop failed, proceeding without it: %s", exc)
            tool_records = []

    payload = generate_structured(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=_build_prompt(situation, matches, employee_id, tool_records),
        response_model=GeminiAnalysisPayload,
    )

    company_data_used = (
        [_to_company_data_source(r, employee_id, employee_name) for r in tool_records if not r.error]
        if employee_id is not None
        else []
    )

    return AnalysisResult(
        **payload.model_dump(),
        knowledge_used=[_to_knowledge_source(m) for m in matches],
        company_data_used=company_data_used,
    )
