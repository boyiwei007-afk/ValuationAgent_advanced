from __future__ import annotations

import os

from valuationagent.finance.team_model import FinanceTeamModel


def create_financial_model():
    """Select the production engine; synthetic demos require request.mode=demo."""
    selected = os.getenv("VALUATION_FINANCE_MODEL", "team").strip().lower()
    if selected == "team":
        return FinanceTeamModel()
    raise ValueError(
        "VALUATION_FINANCE_MODEL must be team; legacy engine switching is removed"
    )
