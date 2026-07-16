from __future__ import annotations

import numpy as np
import pandas as pd

from fund_agent.config import FundSpec
from fund_agent.fund_analysis import analyze_funds, build_research_signals
from fund_agent.skills import load_skill_definitions
from fund_agent.user_profile import MarketActionSettings


def _analysis(target_weight: float = 0.7):
    index = pd.date_range("2025-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {
            "000001": np.cumprod(np.full(100, 1.002)),
            "000002": np.cumprod(np.full(100, 0.998)),
        },
        index=index,
    )
    universe = (
        FundSpec("000001", "基金A", "equity", True, "research_fund"),
        FundSpec("000002", "基金B", "bond", False, "research_fund"),
    )
    return analyze_funds(
        prices=prices,
        fund_universe=universe,
        target_weights=pd.Series({"000001": target_weight, "000002": 1 - target_weight}),
        portfolio=pd.DataFrame([{"code": "000001", "current_value_yuan": 1000}]),
        forecast=None,
        skill_definitions=load_skill_definitions(),
        enabled_skill_ids=["momentum_confirmation", "drawdown_guard"],
        action_settings=MarketActionSettings(
            decision_threshold=0.10,
            minimum_confidence=0.20,
            risk_exit_threshold=0.65,
        ),
    )


def test_analyzes_every_fund_with_neutral_research_signals() -> None:
    result = _analysis()

    assert result.table.index.tolist() == ["000001", "000002"]
    assert result.table.loc["000001", "signal"] == "偏强"
    assert result.table.loc["000002", "signal"] in {"偏弱", "风险退出"}
    assert "suggested_weight_change" not in result.table
    assert "estimated_amount_yuan" not in result.table
    signals = build_research_signals(result.table)
    assert {item["signal"] for item in signals} <= {"strong", "weak", "risk_exit"}
    assert all("action" not in item for item in signals)


def test_reference_target_weight_does_not_control_signal() -> None:
    high_target = _analysis(0.9)
    low_target = _analysis(0.0)

    assert high_target.table.loc["000001", "reference_target_weight"] == 0.9
    assert low_target.table.loc["000001", "reference_target_weight"] == 0.0
    assert high_target.table.loc["000001", "signal"] == low_target.table.loc["000001", "signal"]


def test_unheld_fund_still_receives_research_signal_without_trade_fields() -> None:
    result = _analysis()

    assert result.table.loc["000002", "current_weight"] == 0.0
    assert result.table.loc["000002", "signal"] in {"偏弱", "风险退出"}
    assert any("仅用于研究观察" in risk for risk in result.table.loc["000002", "risks"])
