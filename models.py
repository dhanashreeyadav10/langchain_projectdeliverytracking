from __future__ import annotations
import pandas as pd

# --- UTILIZATION ---
def utilization_model(df: pd.DataFrame) -> pd.DataFrame:
    util = df.groupby("employee_id")["hours_logged"].sum().reset_index()
    util["utilization_pct"] = (util["hours_logged"] / 160) * 100
    meta_cols = [c for c in ["employee_name", "department"] if c in df.columns]
    if meta_cols:
        meta = df[["employee_id", *meta_cols]].drop_duplicates("employee_id")
        util = util.merge(meta, on="employee_id", how="left")
    return util

# --- DELIVERY RISK (Jira-like) ---
def delivery_risk_model(df: pd.DataFrame) -> pd.DataFrame:
    risk = (
        df.groupby("project_id")
        .agg(
            total_hours=("hours_logged", "sum"),
            open_tickets=("ticket_status", lambda x: (x != "Done").sum()),
            high_priority=("priority", lambda x: (x.isin(["High", "Critical"]).sum())),
        )
        .reset_index()
    )
    risk["risk_flag"] = ((risk["open_tickets"] > 5) | (risk["high_priority"] > 3)).astype(int)
    return risk

# --- COST & MARGIN ---
def cost_margin_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delivery_cost"] = df["hours_logged"] * df["cost_per_hour"]
    df["revenue"] = df["hours_logged"] * df["billing_rate"] * df["billable"]

    summary = (
        df.groupby("project_id")
        .agg(
            total_cost=("delivery_cost", "sum"),
            total_revenue=("revenue", "sum"),
            planned_hours=("planned_hours", "max"),
            avg_cost_per_hour=("cost_per_hour", "mean"),
        )
        .reset_index()
    )
    summary["planned_cost"] = summary["planned_hours"] * summary["avg_cost_per_hour"]
    summary["margin"] = summary["total_revenue"] - summary["total_cost"]
    summary["cost_overrun"] = (summary["total_cost"] > summary["planned_cost"]).astype(int)
    return summary

# --- HR SIGNALS ---
def hr_health_model(df: pd.DataFrame) -> pd.DataFrame:
    hr = (
        df.groupby("employee_id")
        .agg(
            avg_attendance=("attendance_pct", "mean"),
            total_leaves=("leave_days", "sum"),
            avg_rating=("performance_rating", "mean"),
        )
        .reset_index()
    )
    hr["hr_risk"] = ((hr["avg_attendance"] < 90) | (hr["avg_rating"] < 3.5)).astype(int)
    return hr
