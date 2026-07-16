from __future__ import annotations

from pathlib import Path

import pandas as pd

from fund_agent.streamlit_app import (
    _is_private_user_path,
    _load_historical_run_states,
    _run_overview_row,
    _signal_display_frame,
    _validate_and_save_portfolio,
)


def test_run_overview_calculates_duration_and_audit_counts() -> None:
    row = _run_overview_row(
        {
            "run_id": "daily-1",
            "task_type": "daily_rebalance",
            "scheduled_for": "2025-01-01T14:30:00+08:00",
            "status": "blocked_by_risk",
            "proposed_actions": [{"signal": "risk_exit"}],
            "risk_checks": {"hard_violations": ["schema"], "warnings": ["risk"]},
            "trace": [
                {"timestamp": "2025-01-01T14:30:00+08:00"},
                {"timestamp": "2025-01-01T14:30:12+08:00"},
            ],
        }
    )

    assert row["研究项"] == 1
    assert row["硬风险"] == 1
    assert row["提示"] == 1
    assert row["耗时（秒）"] == 12.0


def test_portfolio_editor_saves_only_allowed_local_data(tmp_path: Path) -> None:
    path = tmp_path / "alice" / "portfolio.csv"
    valid = pd.DataFrame(
        [
            {"code": "51", "name": "基金A", "current_weight": 0.4},
            {"code": "000052", "name": "基金B", "current_weight": 0.5},
        ]
    )

    error = _validate_and_save_portfolio(valid, path)

    assert error is None
    saved = pd.read_csv(path, dtype={"code": str})
    assert saved["code"].tolist() == ["000051", "000052"]
    assert saved.columns.tolist() == [
        "code",
        "name",
        "current_weight",
        "current_value_yuan",
        "notes",
    ]
    assert _is_private_user_path(path)


def test_signal_display_uses_neutral_labels() -> None:
    display = _signal_display_frame(
        [{"signal": "risk_exit", "code": "000051", "signal_score": -0.8}]
    )

    assert display.loc[0, "signal"] == "风险退出"
    assert "action" not in display


def test_historical_runs_are_loaded(tmp_path: Path) -> None:
    from fund_agent.user_profile import initialize_user_workspace, load_user_profile

    profile_path = initialize_user_workspace(tmp_path / "alice", "alice")
    loaded = load_user_profile(profile_path)
    runs = profile_path.parent / "runs"
    for run_id in ["daily-old", "daily-new"]:
        run_dir = runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run_state.json").write_text(
            '{"run_id":"' + run_id + '"}', encoding="utf-8"
        )

    states = _load_historical_run_states(loaded)

    assert {state["run_id"] for state in states} == {"daily-old", "daily-new"}
