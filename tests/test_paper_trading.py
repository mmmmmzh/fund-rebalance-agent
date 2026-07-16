from __future__ import annotations

import numpy as np
import pandas as pd

from fund_agent.paper_trading import (
    load_paper_ledger,
    record_run_recommendations,
    refresh_recommendation_outcomes,
)


def test_records_signals_idempotently_without_execution_fields(tmp_path) -> None:
    index = pd.date_range("2025-01-01", periods=80, freq="B")
    prices = pd.DataFrame({"000001": np.cumprod(np.full(len(index), 1.002))}, index=index)
    prices_path = tmp_path / "prices.csv"
    prices.iloc[:10].to_csv(prices_path)
    ledger_path = tmp_path / "paper_signals.csv"
    state = {
        "task_type": "daily_rebalance",
        "run_id": "daily-001",
        "scheduled_for": "2025-01-14T14:30:00+08:00",
        "status": "awaiting_approval",
        "prices_path": str(prices_path),
        "research_actions": [
            {
                "signal": "strong",
                "code": "000001",
                "name": "测试基金",
                "signal_score": 0.45,
                "signal_confidence": 0.70,
            }
        ],
    }

    assert record_run_recommendations(ledger_path, state) == 1
    assert record_run_recommendations(ledger_path, state) == 0
    ledger = refresh_recommendation_outcomes(ledger_path, prices)

    assert ledger.iloc[0]["signal"] == "strong"
    assert ledger.iloc[0]["signal_return_5d"] > 0
    assert "actual_amount_yuan" not in ledger
    assert "execution_status" not in ledger


def test_ledger_ignores_non_daily_workflows(tmp_path) -> None:
    path = tmp_path / "signals.csv"
    recorded = record_run_recommendations(
        path,
        {"task_type": "weekly_screen", "proposed_actions": [{"signal": "strong"}]},
    )
    assert recorded == 0
    assert not path.exists()
    assert load_paper_ledger(path).empty
