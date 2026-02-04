from __future__ import annotations
import pandas as pd

class UtilizationAgent:
    def run(self, util_df: pd.DataFrame) -> pd.DataFrame:
        return util_df[util_df["utilization_pct"] < 60]

class DeliveryRiskAgent:
    def run(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        return risk_df[risk_df["risk_flag"] == 1]

class CostMarginAgent:
    def run(self, cost_df: pd.DataFrame) -> pd.DataFrame:
        return cost_df[cost_df["margin"] < 0]

class HRRiskAgent:
    def run(self, hr_df: pd.DataFrame) -> pd.DataFrame:
        return hr_df[hr_df["hr_risk"] == 1]


