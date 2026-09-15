"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_SITUATION_LENGTH = 4000;

type KnowledgeSource = {
  title: string;
  excerpt: string;
  relevance_score: number;
};

type CompanyDataSource = {
  tool: string;
  employee_id: string;
  employee_name: string;
  summary: string;
};

type AnalysisResult = {
  situation_summary: string;
  situation_type: string;
  confidence: number;
  observed_facts: string[];
  assumptions_to_avoid: string[];
  missing_context: string[];
  clarifying_questions: string[];
  recommended_actions: string[];
  conversation_plan: string[];
  risks: string[];
  reasoning_basis: string;
  knowledge_used: KnowledgeSource[];
  company_data_used: CompanyDataSource[];
};

const COMPANY_TOOL_LABELS: Record<string, string> = {
  get_employee: "Employee Record",
  get_jira_activity: "Jira Activity",
  get_github_activity: "GitHub Activity",
  get_one_on_ones: "1:1 Notes",
  get_feedback_history: "Feedback History",
};

function companyToolLabel(tool: string): string {
  return COMPANY_TOOL_LABELS[tool] ?? tool;
}

type Status = "idle" | "loading" | "success" | "error";

function extractErrorMessage(body: unknown, fallback: string): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (typeof item === "object" && item !== null && "msg" in item ? String((item as { msg: unknown }).msg) : null))
        .filter((msg): msg is string => Boolean(msg));
      if (messages.length > 0) return messages.join(" ");
    }
  }
  return fallback;
}

function situationTypeLabel(type: string): string {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const tone =
    confidence >= 0.7
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : confidence >= 0.4
        ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
        : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300";

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium ${tone}`}>
      Confidence: {pct}%
    </span>
  );
}

function ResultSection({
  title,
  items,
  emptyLabel,
  ordered = false,
  tone = "default",
}: {
  title: string;
  items: string[];
  emptyLabel: string;
  ordered?: boolean;
  tone?: "default" | "warning";
}) {
  const ListTag = ordered ? "ol" : "ul";

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <h3
        className={`mb-3 text-sm font-semibold uppercase tracking-wide ${
          tone === "warning" ? "text-amber-700 dark:text-amber-400" : "text-zinc-500 dark:text-zinc-400"
        }`}
      >
        {title}
      </h3>
      {items.length === 0 ? (
        <p className="text-sm text-zinc-400 dark:text-zinc-500">{emptyLabel}</p>
      ) : (
        <ListTag
          className={`space-y-2 text-sm text-zinc-800 dark:text-zinc-200 ${
            ordered ? "list-decimal pl-5" : "list-disc pl-5"
          }`}
        >
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ListTag>
      )}
    </div>
  );
}

function ManagerInputSection({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <h3 className="mb-1 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Manager&apos;s Input
      </h3>
      <p className="mb-3 text-xs text-zinc-400 dark:text-zinc-500">
        Exactly what you described — everything below is built from this, general guidance, and
        (when an employee is named) company evidence retrieved for them.
      </p>
      <blockquote className="whitespace-pre-wrap border-l-2 border-zinc-300 pl-3 text-sm text-zinc-700 dark:border-zinc-700 dark:text-zinc-300">
        {text}
      </blockquote>
    </div>
  );
}

function KnowledgeUsedSection({ sources }: { sources: KnowledgeSource[] }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <h3 className="mb-1 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Management Guidance
      </h3>
      <p className="mb-3 text-xs text-zinc-400 dark:text-zinc-500">
        General management guidance retrieved for this situation — not facts about your employee,
        and not official company policy.
      </p>
      {sources.length === 0 ? (
        <p className="text-sm text-zinc-400 dark:text-zinc-500">
          No sufficiently relevant guidance was found in the knowledge base for this situation.
          The analysis above relies on the situation alone.
        </p>
      ) : (
        <ul className="space-y-3">
          {sources.map((source, i) => (
            <li
              key={i}
              className="rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-900"
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                  {source.title}
                </span>
                <span className="shrink-0 rounded-full bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  {Math.round(source.relevance_score * 100)}% relevant
                </span>
              </div>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">{source.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function CompanyEvidenceSection({ sources }: { sources: CompanyDataSource[] }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <h3 className="mb-1 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Company Evidence
      </h3>
      <p className="mb-3 text-xs text-zinc-400 dark:text-zinc-500">
        Records retrieved directly from company systems (Jira, GitHub, 1:1 notes, feedback
        history) for the employee named in your situation — observed evidence, not something you
        typed and not general guidance.
      </p>
      {sources.length === 0 ? (
        <p className="text-sm text-zinc-400 dark:text-zinc-500">
          No company or employee evidence was retrieved for this analysis. This happens either
          because no specific employee was named in the situation, or because the tools found
          nothing usable to check. The analysis above relies on the situation and general
          guidance alone.
        </p>
      ) : (
        <ul className="space-y-3">
          {sources.map((source, i) => {
            const isEmpty = source.summary.toLowerCase().startsWith("no records found");
            return (
              <li
                key={i}
                className="rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-900"
              >
                <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                    {companyToolLabel(source.tool)}
                  </span>
                  <span className="shrink-0 rounded-full bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                    {source.employee_name}
                  </span>
                </div>
                <p
                  className={
                    isEmpty
                      ? "text-sm italic text-zinc-400 dark:text-zinc-500"
                      : "text-sm text-zinc-600 dark:text-zinc-400"
                  }
                >
                  {source.summary}
                </p>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-4 text-xs italic text-zinc-400 dark:text-zinc-500">
        Company data shown here is simulated demo data.
      </p>
    </div>
  );
}

export default function Home() {
  const [situation, setSituation] = useState("");
  const [submittedSituation, setSubmittedSituation] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const trimmed = situation.trim();
  const isDisabled = status === "loading" || trimmed.length === 0 || trimmed.length > MAX_SITUATION_LENGTH;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (isDisabled) return;

    setStatus("loading");
    setErrorMessage("");
    setResult(null);
    setSubmittedSituation(trimmed);

    let res: Response;
    try {
      res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ situation: trimmed }),
      });
    } catch {
      setErrorMessage("Could not reach the ManagerLens backend. Make sure it's running and try again.");
      setStatus("error");
      return;
    }

    const body = await res.json().catch(() => null);

    if (!res.ok) {
      setErrorMessage(extractErrorMessage(body, `Request failed with status ${res.status}.`));
      setStatus("error");
      return;
    }

    setResult(body as AnalysisResult);
    setStatus("success");
  }

  return (
    <div className="min-h-screen bg-zinc-50 px-4 py-10 font-sans dark:bg-black sm:px-8">
      <div className="mx-auto flex max-w-3xl flex-col gap-8">
        <header className="text-center">
          <div className="mb-2 flex justify-end">
            <Link
              href="/evaluation"
              className="text-xs font-medium text-zinc-500 underline hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
            >
              Evaluation Dashboard →
            </Link>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            ManagerLens
          </h1>
          <p className="mt-2 text-zinc-600 dark:text-zinc-400">
            Describe a workplace situation. ManagerLens separates what you actually observed from
            what you might be assuming, and gives you grounded next steps — it will not diagnose
            your employee or invent facts.
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950"
        >
          <label htmlFor="situation" className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
            What&apos;s going on?
          </label>
          <textarea
            id="situation"
            value={situation}
            onChange={(e) => setSituation(e.target.value)}
            rows={5}
            maxLength={MAX_SITUATION_LENGTH}
            placeholder="e.g. My engineer has missed three deadlines and has become quiet during team meetings. I am worried they are disengaged."
            className="w-full resize-y rounded-md border border-zinc-300 bg-white p-3 text-sm text-zinc-900 outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
          />
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-zinc-400">
              {trimmed.length}/{MAX_SITUATION_LENGTH}
            </span>
            <button
              type="submit"
              disabled={isDisabled}
              className="rounded-md bg-zinc-900 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
            >
              {status === "loading" ? "Analyzing…" : "Analyze Situation"}
            </button>
          </div>
        </form>

        {status === "loading" && (
          <div className="flex items-center justify-center gap-3 rounded-lg border border-zinc-200 bg-white p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-600 dark:border-zinc-700 dark:border-t-zinc-300" />
            Analyzing your situation…
          </div>
        )}

        {status === "error" && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-5 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
            <p className="font-medium">Analysis failed</p>
            <p className="mt-1">{errorMessage}</p>
          </div>
        )}

        {status === "success" && result && (
          <div className="flex flex-col gap-5">
            <ManagerInputSection text={submittedSituation} />

            <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                ManagerLens Analysis
              </h3>
              <div className="mb-3 flex flex-wrap items-center gap-3">
                <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200">
                  {situationTypeLabel(result.situation_type)}
                </span>
                <ConfidenceBadge confidence={result.confidence} />
              </div>
              <p className="text-zinc-800 dark:text-zinc-200">{result.situation_summary}</p>
              <p className="mt-3 text-xs italic text-zinc-500 dark:text-zinc-500">{result.reasoning_basis}</p>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <ResultSection
                title="Observed Facts"
                items={result.observed_facts}
                emptyLabel="No explicit facts were identified."
              />
              <ResultSection
                title="Assumptions to Avoid"
                items={result.assumptions_to_avoid}
                emptyLabel="No risky assumptions flagged."
                tone="warning"
              />
              <ResultSection
                title="Missing Context"
                items={result.missing_context}
                emptyLabel="No significant gaps identified."
              />
              <ResultSection
                title="Clarifying Questions"
                items={result.clarifying_questions}
                emptyLabel="No clarifying questions suggested."
              />
            </div>

            <ResultSection
              title="Recommended Actions"
              items={result.recommended_actions}
              emptyLabel="No actions recommended."
              ordered
            />
            <ResultSection
              title="Conversation Plan"
              items={result.conversation_plan}
              emptyLabel="No conversation plan generated."
              ordered
            />
            <ResultSection
              title="Risks"
              items={result.risks}
              emptyLabel="No risks flagged."
            />

            <CompanyEvidenceSection sources={result.company_data_used} />
            <KnowledgeUsedSection sources={result.knowledge_used} />
          </div>
        )}
      </div>
    </div>
  );
}
