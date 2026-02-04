# -*- coding: utf-8 -*-
"""
Conversational QA: routes the user's question to a delivery theme,
and returns either (a) a real data table when the user asks for a list,
or (b) an advisory answer via the Groq SDK-backed explain_insight().

Return signature:
    (answer_text: str, table_df: Optional[pd.DataFrame])

- If table_df is not None, render it in Streamlit and optionally offer a CSV download.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple
import pandas as pd
from llm import explain_insight

__all__ = ["route_and_summarize"]

# --------------------------
# Combined Prompt Template (advisory mode)
# --------------------------
PROMPT_TEMPLATE = """
You are an enterprise delivery intelligence advisor. Your audience may range from delivery managers to the CXO team.
Synthesize insights that are data-grounded, practical, and business-aware.

INPUTS
- User Question:
{question}

- Context Summary:
{summary}

- (Optional) Metrics Snapshot (include only if provided):
{metrics}

OBJECTIVE
Combine the strengths of multiple advisory styles:
• Executive Consulting (crisp, board-ready)
• Root-Cause + Recommendations
• Data-Driven Delivery Analytics
• C‑Suite Advisory (strategy + outcomes)
• Issue → Impact → Intervention
• Delivery Coaching (clear next actions)
• Risk-First Thinking (severity/urgency)
• Balanced Scorecard (People, Process, Project Execution, Financials)

RESPONSE FORMAT (use concise bullets; include sections only if they add value)
1) Executive Insight (1–2 lines)
   - The one-liner an executive should remember.

2) Root Causes
   - Probable drivers and how they relate to the question and summary.

3) Risks & Severity (if any)
   - Key delivery/quality/schedule/people/financial risks.
   - Indicate severity and urgency (e.g., High/Med/Low).

4) Business & Delivery Impact
   - Effects on margin, cost overrun, revenue leakage, utilization, velocity, customer commitments, SLAs, or timelines.

5) Recommendations (prioritized 2–5 actions)
   - {action_owner_hint}
   - Include: owner/function, timeframe (e.g., 1–2 weeks), and success metric.

6) What to Monitor Next
   - Leading indicators or checkpoints to validate improvement.

7) Balanced Scorecard View
   - People: {balanced_people_hint}
   - Process:
   - Project Execution:
   - Financial Health:

8) Assumptions / Data Gaps
   - Call out any missing context or data quality risks influencing your answer.

WRITING GUIDELINES
- Be specific, actionable, and minimally verbose.
- Prefer bullet points. Avoid generic advice.
- Where possible, tie recommendations to metrics or thresholds (e.g., utilization < 60%, margin < 0).
- If context is too thin, state the assumption and propose what extra data to fetch.

OUTPUT
Return only the advisory content in the structure above.
"""

# --------------------------
# Helpers
# --------------------------
def _compact_markdown_table(
    df: pd.DataFrame, max_rows: int = 50, cols: Optional[list] = None
) -> str:
    """
    Convert a slice of DataFrame to a compact Markdown table (for the answer_text).
    Defaults to up to 50 rows to keep the message readable; the full table_df
    is returned separately for Streamlit to render and download.
    """
    if df is None or df.empty:
        return ""
    if cols is None:
        cols = list(df.columns)
    slim = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()

    # Pretty formatting for key numeric fields
    for c in ["utilization_pct", "margin"]:
        if c in slim.columns:
            slim[c] = pd.to_numeric(slim[c], errors="coerce").round(2)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in slim.iterrows():
        vals = [(("" if pd.isna(v) else str(v))) for v in r.tolist()]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


# --------------------------
# Main QA entry
# --------------------------
def route_and_summarize(
    question: str,
    util_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    include_metrics: bool = True,
    metrics_rows: int = 50,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Detect user intent. If the question asks for a LIST/SHOW/WHO, return actual rows.
    Otherwise, return an advisory LLM answer.

    Returns:
        answer_text (str), table_df (Optional[pd.DataFrame])
    """
    q_raw = question or ""
    q = q_raw.strip().lower()

    wants_list = any(
        k in q
        for k in [
            "list", "show", "display", "give me", "who are", "names of",
            "export", "download", "table of", "print", "return rows", "records"
        ]
    )

    # Default outputs
    answer_text: str = ""
    table_df: Optional[pd.DataFrame] = None

    # ----------------------
    # Routing by domain
    # ----------------------
    is_util = ("utilization" in q) or ("underutilized" in q) or ("underutilised" in q) or ("bench" in q) or ("under utilize" in q) or ("under utilise" in q)
    is_risk = ("risk" in q) or ("delay" in q) or ("blocker" in q)
    is_cost = ("cost" in q) or ("margin" in q) or ("loss" in q) or ("overrun" in q)
    is_hr   = ("hr" in q) or ("attrition" in q) or ("attendance" in q) or ("performance" in q)

    # ----------------------
    # LIST MODE: return real rows (no LLM needed)
    # ----------------------
    if wants_list and is_util:
        # Underutilized employees
        df = util_df.copy()
        df["utilization_pct"] = pd.to_numeric(df.get("utilization_pct", 100), errors="coerce")
        df = df[df["utilization_pct"] < 60].copy()
        cols = [c for c in ["employee_id", "employee_name", "department", "utilization_pct"] if c in df.columns]
        if not cols:
            cols = list(df.columns)
        df = df[cols].sort_values(by="utilization_pct", ascending=True, na_position="last")
        df["utilization_pct"] = df["utilization_pct"].round(2)
        table_df = df.reset_index(drop=True)

        table_md = _compact_markdown_table(table_df, max_rows=metrics_rows, cols=cols)
        header = f"### Underutilized employees ({len(table_df)})\n"
        note = "_Showing first {} rows below. Use the download button to get the full list._".format(
            min(metrics_rows, len(table_df))
        ) if len(table_df) > metrics_rows else ""
        answer_text = f"{header}{note}\n\n{table_md}"
        return answer_text, table_df

    if wants_list and is_risk:
        df = risk_df.copy()
        df = df[df.get("risk_flag", 0) == 1].copy()
        cols = [c for c in ["project_id", "project_name", "client_name", "open_tickets", "high_priority"] if c in df.columns]
        if not cols:
            cols = list(df.columns)
        sort_cols = [c for c in ["high_priority", "open_tickets"] if c in df.columns]
        df = df[cols].sort_values(by=sort_cols, ascending=False) if sort_cols else df[cols]
        table_df = df.reset_index(drop=True)

        table_md = _compact_markdown_table(table_df, max_rows=metrics_rows, cols=cols)
        header = f"### High‑risk projects ({len(table_df)})\n"
        note = "_Showing first {} rows below._".format(min(metrics_rows, len(table_df))) if len(table_df) > metrics_rows else ""
        answer_text = f"{header}{note}\n\n{table_md}"
        return answer_text, table_df

    if wants_list and is_cost:
        df = cost_df.copy()
        df["margin"] = pd.to_numeric(df.get("margin", 0), errors="coerce")
        df = df[df["margin"] < 0].copy()
        cols = [c for c in ["project_id", "project_name", "client_name", "margin"] if c in df.columns]
        if not cols:
            cols = list(df.columns)
        df = df[cols].sort_values(by="margin", ascending=True, na_position="last")
        table_df = df.reset_index(drop=True)

        table_md = _compact_markdown_table(table_df, max_rows=metrics_rows, cols=cols)
        header = f"### Loss‑making / negative‑margin projects ({len(table_df)})\n"
        note = "_Showing first {} rows below._".format(min(metrics_rows, len(table_df))) if len(table_df) > metrics_rows else ""
        answer_text = f"{header}{note}\n\n{table_md}"
        return answer_text, table_df

    if wants_list and is_hr:
        df = hr_df.copy()
        df = df[df.get("hr_risk", 0) == 1].copy()
        cols = [c for c in ["employee_id", "employee_name", "department", "avg_attendance", "avg_rating", "total_leaves"] if c in df.columns]
        if not cols:
            cols = list(df.columns)
        sort_cols = [c for c in ["avg_attendance", "avg_rating"] if c in df.columns]
        df = df[cols].sort_values(by=sort_cols, ascending=True) if sort_cols else df[cols]
        table_df = df.reset_index(drop=True)

        table_md = _compact_markdown_table(table_df, max_rows=metrics_rows, cols=cols)
        header = f"### Employees with HR‑risk indicators ({len(table_df)})\n"
        note = "_Showing first {} rows below._".format(min(metrics_rows, len(table_df))) if len(table_df) > metrics_rows else ""
        answer_text = f"{header}{note}\n\n{table_md}"
        return answer_text, table_df

    # ----------------------
    # ADVISORY MODE (LLM)
    # ----------------------
    summary = "Overall delivery, HR, and financial health overview."
    themed_df = None
    if is_util:
        themed_df = util_df.loc[pd.to_numeric(util_df.get("utilization_pct", 100), errors="coerce") < 60]
        summary = f"{len(themed_df)} employees are underutilized."
    elif is_risk:
        themed_df = risk_df.loc[(risk_df.get("risk_flag", 0) == 1)]
        summary = f"{len(themed_df)} projects are at delivery risk."
    elif is_cost:
        themed_df = cost_df.loc[pd.to_numeric(cost_df.get("margin", 0), errors="coerce") < 0]
        summary = f"{len(themed_df)} projects are loss-making."
    elif is_hr:
        themed_df = hr_df.loc[(hr_df.get("hr_risk", 0) == 1)]
        summary = f"{len(themed_df)} employees show HR risk indicators."

    metrics_snippet = ""
    if themed_df is not None and include_metrics and not themed_df.empty:
        if is_util:
            cols = [c for c in ["employee_id", "employee_name", "department", "utilization_pct"] if c in themed_df.columns]
        elif is_risk:
            cols = [c for c in ["project_id", "project_name", "client_name", "open_tickets", "high_priority"] if c in themed_df.columns]
        elif is_cost:
            cols = [c for c in ["project_id", "project_name", "client_name", "margin"] if c in themed_df.columns]
        else:
            cols = list(themed_df.columns[:6])
        metrics_snippet = _compact_markdown_table(themed_df, max_rows=min(5, metrics_rows), cols=cols)

    action_owner_hint = "Assign each action to a single accountable owner and a target date; avoid shared ownership."
    balanced_people_hint = "Utilization, attendance, performance, and attrition risk."

    prompt = PROMPT_TEMPLATE.format(
        question=question,
        summary=summary,
        metrics=metrics_snippet,
        action_owner_hint=action_owner_hint,
        balanced_people_hint=balanced_people_hint,
    )

    if not os.getenv("GROQ_API_KEY"):
        answer_text = (
            "❌ LLM unavailable: GROQ_API_KEY is not set for this process.\n\n"
            f"**Question:** {question}\n**Context Summary:** {summary}\n\n"
            "Set the key via `.streamlit/secrets.toml` or environment and retry."
        )
        return answer_text, None

    try:
        answer_text = explain_insight(prompt)
        return answer_text, None
    except Exception as e:
        answer_text = (
            f"❌ LLM call failed: {e}\n\n"
            f"**Question:** {question}\n**Context Summary:** {summary}\n"
            "Please check the LLM Health Check in the sidebar."
        )
        return answer_text, None






