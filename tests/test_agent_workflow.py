from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from fund_agent.agent_workflow import (
    _portfolio_input_error,
    resume_agent_workflow,
    run_agent_workflow,
    workflow_status,
)
from fund_agent.user_profile import initialize_user_workspace


SCHEDULED = datetime(2025, 3, 28, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def _profile(tmp_path: Path) -> Path:
    profile_path = initialize_user_workspace(tmp_path / "user", "test-user")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["start_date"] = "2024-01-01"
    payload["lookback_days"] = 40
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile_path


def test_daily_workflow_interrupts_and_can_be_rejected(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["market_actions"] = {
        "decision_threshold": 0.05,
        "minimum_confidence": 0.10,
        "risk_exit_threshold": 0.65,
    }
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    universe = pd.read_csv(profile_path.parent / "universe.csv", dtype={"code": str})
    pd.DataFrame(
        [
            {"code": row["code"], "name": row["name"], "current_value_yuan": 1000}
            for _, row in universe.iterrows()
        ]
    ).to_csv(profile_path.parent / "portfolio.csv", index=False)
    result = run_agent_workflow(
        profile_path,
        "daily_rebalance",
        scheduled_for=SCHEDULED,
        run_id="daily-test",
    )

    assert result["status"] == "awaiting_approval"
    assert result["interrupts"]
    assert Path(result["analysis_report_path"]).exists()
    assert len(result["fund_analyses"]) == len(universe)
    assert result["research_actions"]
    assert len(result["research_actions"]) >= len(result["proposed_actions"])
    assert result["portfolio_summary"]["fund_count"] == len(universe)
    assert "逐基金净值技术面与市场环境判断" in Path(result["analysis_report_path"]).read_text(
        encoding="utf-8"
    )
    assert [event["agent"] for event in result["trace"]] == [
        "ProfileAgent",
        "DataCollectionAgent",
        "MarketResearchAgent",
        "PortfolioAgent",
        "RiskReviewAgent",
        "ReviewAgent",
        "PaperValidationLedger",
    ]
    assert (profile_path.parent / "paper_trades.csv").stat().st_size > 0
    assert result["risk_checks"]["review_signal_count"] > 0
    assert workflow_status(profile_path, "daily-test")["next_nodes"] == [
        "human_approval_agent"
    ]

    completed = resume_agent_workflow(profile_path, "daily-test", "reject", "不接受本次建议")

    assert completed["status"] == "rejected"
    assert completed["approval"]["feedback"] == "不接受本次建议"
    assert completed["trace"][-2]["agent"] == "HumanApprovalAgent"
    assert completed["trace"][-1]["agent"] == "AuditAgent"
    run_state = json.loads(
        (Path(completed["run_dir"]) / "run_state.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "rejected"
    assert run_state["trace"][-1]["agent"] == "AuditAgent"


def test_weekly_approval_updates_investable_universe(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path)
    result = run_agent_workflow(
        profile_path,
        "weekly_screen",
        scheduled_for=SCHEDULED,
        run_id="weekly-test",
    )

    assert result["status"] == "awaiting_approval"
    assert len(result["proposed_actions"]) == 2
    watchlist = pd.read_csv(profile_path.parent / "candidate_watchlist.csv", dtype={"code": str})
    assert len(watchlist) == 5

    completed = resume_agent_workflow(profile_path, "weekly-test", "approve")

    assert completed["status"] == "approved"
    investable = pd.read_csv(
        profile_path.parent / "investable_universe.csv",
        dtype={"code": str},
    )
    approved = investable[investable["status"] == "human_approved"]
    assert len(approved) == 2
    assert approved["code"].str.startswith("9").all()


def test_monthly_workflow_completes_without_approval(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path)
    result = run_agent_workflow(
        profile_path,
        "monthly_research",
        scheduled_for=SCHEDULED,
        run_id="monthly-test",
    )

    assert result["status"] == "completed"
    assert "interrupts" not in result
    assert result["approval_required"] is False
    assert Path(result["monthly_report_path"]).exists()
    assert [event["agent"] for event in result["trace"]] == [
        "ProfileAgent",
        "IndustryResearchAgent",
        "ReviewAgent",
        "AuditAgent",
    ]


def test_portfolio_input_rejects_zero_or_missing_positions() -> None:
    frame = pd.DataFrame(
        [
            {"code": "000001", "current_value_yuan": 1000},
            {"code": "000002", "current_value_yuan": None, "current_weight": 0.0},
        ]
    )

    error = _portfolio_input_error(frame)

    assert error is not None
    assert "000002" in error


@pytest.mark.parametrize("run_id", ["../escape", "..\\escape", "/tmp/escape", "C:\\escape"])
def test_workflow_rejects_path_like_run_ids(tmp_path: Path, run_id: str) -> None:
    profile_path = _profile(tmp_path)

    with pytest.raises(ValueError, match="run_id"):
        run_agent_workflow(
            profile_path,
            "monthly_research",
            scheduled_for=SCHEDULED,
            run_id=run_id,
        )
