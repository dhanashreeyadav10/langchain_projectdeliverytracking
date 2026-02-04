# # from __future__ import annotations
# from typing import Optional, TypedDict
# import pandas as pd
# from langgraph.graph import StateGraph, END
# from models import utilization_model, delivery_risk_model, cost_margin_model, hr_health_model
# from agents import UtilizationAgent, DeliveryRiskAgent, CostMarginAgent, HRRiskAgent

# from agents import delivery_agent


# def run_delivery_graph(question, context):
#     return delivery_agent(question, context)


# class AppState(TypedDict, total=False):
#     raw_df: pd.DataFrame
#     util_df: pd.DataFrame
#     risk_df: pd.DataFrame
#     cost_df: pd.DataFrame
#     hr_df: pd.DataFrame
#     low_util: pd.DataFrame
#     risk_projects: pd.DataFrame
#     loss_projects: pd.DataFrame
#     hr_risks: pd.DataFrame
#     use_llm: bool
#     llm_summary: Optional[str]

# # --- Nodes ---

# def compute_features(state: AppState) -> AppState:
#     df = state["raw_df"]
#     state["util_df"] = utilization_model(df)
#     state["risk_df"] = delivery_risk_model(df)
#     state["cost_df"] = cost_margin_model(df)
#     state["hr_df"] = hr_health_model(df)
#     return state

# def run_util_agent(state: AppState) -> AppState:
#     agent = UtilizationAgent()
#     state["low_util"] = agent.run(state["util_df"])
#     return state

# def run_risk_agent(state: AppState) -> AppState:
#     agent = DeliveryRiskAgent()
#     state["risk_projects"] = agent.run(state["risk_df"])
#     return state

# def run_cost_agent(state: AppState) -> AppState:
#     agent = CostMarginAgent()
#     state["loss_projects"] = agent.run(state["cost_df"])
#     return state

# def run_hr_agent(state: AppState) -> AppState:
#     agent = HRRiskAgent()
#     state["hr_risks"] = agent.run(state["hr_df"])
#     return state

# def summarize_node(state: AppState) -> AppState:
#     if state.get("use_llm"):
#         low = len(state.get("low_util", []))
#         risky = len(state.get("risk_projects", []))
#         loss = len(state.get("loss_projects", []))
#         hr = len(state.get("hr_risks", []))
#         prompt = f"""
# Enterprise Delivery Intelligence Summary:
# Underutilized Employees: {low}
# High Delivery Risk Projects: {risky}
# Loss-Making Projects: {loss}
# HR Risk Employees: {hr}
# Provide:
# - Executive insights
# - Root causes
# - Actionable recommendations
#         """
#         # Lazy import to avoid import-time errors
#         try:
#             from .llm import explain_insight
#             state["llm_summary"] = explain_insight(prompt)
#         except Exception as e:
#             state["llm_summary"] = f"❌ LLM unavailable: {e}"
#     return state

# def build_graph():
#     g = StateGraph(AppState)
#     g.add_node("features", compute_features)
#     g.add_node("util", run_util_agent)
#     g.add_node("risk", run_risk_agent)
#     g.add_node("cost", run_cost_agent)
#     g.add_node("hr", run_hr_agent)
#     g.add_node("summarize", summarize_node)

#     g.set_entry_point("features")
#     g.add_edge("features", "util")
#     g.add_edge("util", "risk")
#     g.add_edge("risk", "cost")
#     g.add_edge("cost", "hr")
#     g.add_edge("hr", "summarize")
#     g.add_edge("summarize", END)
#     return g.compile()

from __future__ import annotations
from typing import Optional, TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END

# ✅ absolute imports for repo modules
from models import utilization_model, delivery_risk_model, cost_margin_model, hr_health_model
from agents import UtilizationAgent, DeliveryRiskAgent, CostMarginAgent, HRRiskAgent

class AppState(TypedDict, total=False):
    raw_df: pd.DataFrame
    util_df: pd.DataFrame
    risk_df: pd.DataFrame
    cost_df: pd.DataFrame
    hr_df: pd.DataFrame
    low_util: pd.DataFrame
    risk_projects: pd.DataFrame
    loss_projects: pd.DataFrame
    hr_risks: pd.DataFrame
    use_llm: bool
    llm_summary: Optional[str]

def compute_features(state: AppState) -> AppState:
    df = state["raw_df"]
    state["util_df"] = utilization_model(df)
    state["risk_df"] = delivery_risk_model(df)
    state["cost_df"] = cost_margin_model(df)
    state["hr_df"] = hr_health_model(df)
    return state

def run_util_agent(state: AppState) -> AppState:
    agent = UtilizationAgent()
    state["low_util"] = agent.run(state["util_df"])
    return state

def run_risk_agent(state: AppState) -> AppState:
    agent = DeliveryRiskAgent()
    state["risk_projects"] = agent.run(state["risk_df"])
    return state

def run_cost_agent(state: AppState) -> AppState:
    agent = CostMarginAgent()
    state["loss_projects"] = agent.run(state["cost_df"])
    return state

def run_hr_agent(state: AppState) -> AppState:
    agent = HRRiskAgent()
    state["hr_risks"] = agent.run(state["hr_df"])
    return state

def summarize_node(state: AppState) -> AppState:
    if state.get("use_llm"):
        low = len(state.get("low_util", []))
        risky = len(state.get("risk_projects", []))
        loss = len(state.get("loss_projects", []))
        hr = len(state.get("hr_risks", []))
        prompt = f"""
Enterprise Delivery Intelligence Summary:
Underutilized Employees: {low}
High Delivery Risk Projects: {risky}
Loss-Making Projects: {loss}
HR Risk Employees: {hr}
Provide:
- Executive insights
- Root causes
- Actionable recommendations
        """
        # ✅ ABSOLUTE, LAZY import with diagnostic on failure
        try:
            from llm import explain_insight  # absolute import
            state["llm_summary"] = explain_insight(prompt)
        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=1)  # short hint (top line)
            state["llm_summary"] = f"❌ LLM unavailable: {e} | {tb.strip()}"
    return state

def build_graph():
    g = StateGraph(AppState)
    g.add_node("features", compute_features)
    g.add_node("util", run_util_agent)
    g.add_node("risk", run_risk_agent)
    g.add_node("cost", run_cost_agent)
    g.add_node("hr", run_hr_agent)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("features")
    g.add_edge("features", "util")
    g.add_edge("util", "risk")
    g.add_edge("risk", "cost")
    g.add_edge("cost", "hr")
    g.add_edge("hr", "summarize")
    g.add_edge("summarize", END)
    return g.compile()



