# Mock Company Data

**This is entirely simulated data.** It does not represent any real company, team, employee, or
event. "Windmill" is a fictional company name used only for this project. All names
(Jordan Alvarez, Priya Natarajan, Casey Kim, Sam Rivera), issue IDs, pull requests, and notes
below were invented for local development, demos, and the evaluation suite — nothing here is a
real integration with Jira, GitHub, or any other workplace system.

It exists so ManagerLens' future agent/tool phase has realistic-looking evidence to retrieve
and reason over, without needing real company data or real API integrations.

## Files

| File | What it simulates |
|---|---|
| `employees.json` | A basic HR directory record per employee. |
| `jira_activity.json` | Issue tracker history: what was assigned, due, completed, and what blocked it. |
| `github_activity.json` | Pull request history: what was opened, merged, and how much review it got. |
| `one_on_ones.json` | Manager's private notes from 1:1 conversations. |
| `feedback_history.json` | Feedback given about the employee over time, from different sources. |

## The three employees, and why they're each here

- **EMP001 (Jordan Alvarez)** — deliberately designed to *look* like a performance problem at
  first glance (two missed Jira deadlines, an early manager note expressing concern) but whose
  fuller evidence trail (one-on-one notes, peer feedback, a later manager follow-up) shows both
  delays were caused by an external API dependency and a requirements change — not effort or
  skill. This is the important case: an agent that reasons from Jira dates alone would reach the
  wrong conclusion, and should not agree with a manager's "this employee is underperforming"
  assumption without checking the fuller evidence.
- **EMP002 (Priya Natarajan)** — genuinely strong, consistent performance across every data
  source (on-time or early delivery, positive peer/manager feedback, no concerns raised).
- **EMP003 (Casey Kim)** — a recent hire with deliberately sparse history (one one-on-one, one
  merged PR, minimal feedback). Represents the "not enough evidence yet" case — an agent should
  say so rather than render a confident verdict from two data points.

## How this will be used

Phase 1 (this phase) only creates the data — no code reads it yet. A later phase will add tools
that let the manager agent query this data (e.g., "look up recent Jira activity for employee
X") as part of its evidence-gathering, the same way the existing RAG system retrieves guidance
documents. Keeping the data separate from that logic now means the tool layer can be added,
tested, and reviewed independently of the data itself.
