"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type MetricScore = {
  name: string;
  applicable: boolean;
  passed: boolean | null;
  score: number | null;
  critical: boolean;
  details: string;
};

type JudgeVerdict = {
  invents_facts: boolean;
  unsupported_diagnosis: boolean;
  treats_guidance_as_policy: boolean;
  unsafe_certainty: boolean;
  reasonable_overall: boolean;
  rationale: string;
};

type EvaluationResult = {
  test_name: string;
  category: string;
  is_red_team: boolean;
  passed: boolean;
  overall_score: number;
  metrics: MetricScore[];
  judge: JudgeVerdict | null;
  failure_reasons: string[];
  error: string | null;
};

type EvaluationReport = {
  generated_at: string;
  total: number;
  passed: number;
  failed: number;
  overall_score: number;
  metric_averages: Record<string, number>;
  failure_categories: Record<string, number>;
  used_judge: boolean;
  results: EvaluationResult[];
};

type Status = "loading" | "success" | "not_found" | "error";

function StatCard({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "good" | "bad";
}) {
  const valueTone =
    tone === "good"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "bad"
        ? "text-red-600 dark:text-red-400"
        : "text-zinc-900 dark:text-zinc-50";

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <p className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {label}
      </p>
      <p className={`mt-1 text-2xl font-semibold ${valueTone}`}>{value}</p>
    </div>
  );
}

function metricLabel(name: string): string {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function MetricBar({ name, score }: { name: string; score: number }) {
  const pct = Math.round(score * 100);
  const barColor = pct >= 80 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500";

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="text-zinc-700 dark:text-zinc-300">{metricLabel(name)}</span>
        <span className="font-medium text-zinc-900 dark:text-zinc-100">{pct}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function shortenError(error: string, maxLength = 200): string {
  const firstLine = error.split(/\{['"]error['"]/)[0].trim();
  const base = firstLine.length > 10 ? firstLine : error;
  return base.length > maxLength ? `${base.slice(0, maxLength).trim()}…` : base;
}

function FailedCaseCard({ result }: { result: EvaluationResult }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900/50 dark:bg-red-950/30">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-medium text-red-800 dark:text-red-300">
          {result.test_name}
        </span>
        <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700 dark:bg-red-900/50 dark:text-red-300">
          {result.category}
        </span>
        {result.is_red_team && (
          <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs font-medium text-white dark:bg-zinc-200 dark:text-zinc-900">
            red team
          </span>
        )}
        <span className="ml-auto text-xs font-medium text-red-700 dark:text-red-300">
          score: {result.overall_score}
        </span>
      </div>
      {result.error && (
        <p className="mb-1 text-sm text-red-700 dark:text-red-300">
          Generation error: {shortenError(result.error)}
        </p>
      )}
      {!result.error && result.failure_reasons.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-red-800 dark:text-red-200">
          {result.failure_reasons.map((reason, i) => (
            <li key={i}>{shortenError(reason)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function EvaluationDashboard() {
  const [status, setStatus] = useState<Status>("loading");
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [showAllResults, setShowAllResults] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/evaluation`)
      .then(async (res) => {
        if (res.status === 404) {
          setStatus("not_found");
          return;
        }
        if (!res.ok) {
          throw new Error(`Request failed with status ${res.status}`);
        }
        const body = (await res.json()) as EvaluationReport;
        setReport(body);
        setStatus("success");
      })
      .catch((err: Error) => {
        setErrorMessage(err.message);
        setStatus("error");
      });
  }, []);

  const failedResults = report?.results.filter((r) => !r.passed) ?? [];

  return (
    <div className="min-h-screen bg-zinc-50 px-4 py-10 font-sans dark:bg-black sm:px-8">
      <div className="mx-auto flex max-w-4xl flex-col gap-8">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              Evaluation Dashboard
            </h1>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Reliability results for ManagerLens&apos; AI pipeline against the golden + red-team
              dataset.
            </p>
          </div>
          <Link
            href="/"
            className="shrink-0 rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
          >
            ← Analyze
          </Link>
        </header>

        {status === "loading" && (
          <div className="flex items-center justify-center gap-3 rounded-lg border border-zinc-200 bg-white p-6 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-600 dark:border-zinc-700 dark:border-t-zinc-300" />
            Loading evaluation report…
          </div>
        )}

        {status === "not_found" && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
            <p className="font-medium">No evaluation report yet</p>
            <p className="mt-1">
              Run <code className="rounded bg-black/10 px-1.5 py-0.5 dark:bg-white/10">python -m evaluation.runner</code>{" "}
              from the backend directory to generate one.
            </p>
          </div>
        )}

        {status === "error" && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-5 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
            <p className="font-medium">Could not load evaluation report</p>
            <p className="mt-1">{errorMessage}</p>
          </div>
        )}

        {status === "success" && report && (
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatCard label="Total Tests" value={String(report.total)} />
              <StatCard label="Passed" value={String(report.passed)} tone="good" />
              <StatCard label="Failed" value={String(report.failed)} tone={report.failed > 0 ? "bad" : "default"} />
              <StatCard
                label="Overall Score"
                value={`${Math.round(report.overall_score * 100)}%`}
                tone={report.overall_score >= 0.8 ? "good" : report.overall_score >= 0.5 ? "default" : "bad"}
              />
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-500">
              Generated {new Date(report.generated_at).toLocaleString()} · LLM judge{" "}
              {report.used_judge ? "used" : "not used"}
            </p>

            <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                Metric Scores
              </h2>
              {Object.keys(report.metric_averages).length === 0 ? (
                <p className="text-sm text-zinc-400">No applicable metrics recorded.</p>
              ) : (
                <div className="space-y-3">
                  {Object.entries(report.metric_averages).map(([name, score]) => (
                    <MetricBar key={name} name={name} score={score} />
                  ))}
                </div>
              )}
            </div>

            {Object.keys(report.failure_categories).length > 0 && (
              <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Failure Categories
                </h2>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(report.failure_categories).map(([category, count]) => (
                    <span
                      key={category}
                      className="rounded-full bg-red-100 px-3 py-1 text-sm font-medium text-red-800 dark:bg-red-900/40 dark:text-red-300"
                    >
                      {category}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                  Failed Test Cases {failedResults.length > 0 && `(${failedResults.length})`}
                </h2>
                <button
                  onClick={() => setShowAllResults((v) => !v)}
                  className="text-xs font-medium text-zinc-500 underline hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                >
                  {showAllResults ? "Show failed only" : "Show all results"}
                </button>
              </div>

              {failedResults.length === 0 && !showAllResults && (
                <p className="text-sm text-zinc-400 dark:text-zinc-500">
                  All test cases passed. 🎉
                </p>
              )}

              <div className="flex flex-col gap-3">
                {(showAllResults ? report.results : failedResults).map((result) =>
                  result.passed ? (
                    <div
                      key={result.test_name}
                      className="flex items-center justify-between rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm dark:border-emerald-900/50 dark:bg-emerald-950/20"
                    >
                      <span className="font-mono text-emerald-800 dark:text-emerald-300">
                        {result.test_name}
                      </span>
                      <span className="text-emerald-700 dark:text-emerald-400">
                        score: {result.overall_score}
                      </span>
                    </div>
                  ) : (
                    <FailedCaseCard key={result.test_name} result={result} />
                  )
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
