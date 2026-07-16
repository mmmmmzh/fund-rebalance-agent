from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fund_agent.scheduler import load_scheduler_runs, run_due_agent_tasks, scheduled_tasks
from fund_agent.user_profile import initialize_user_workspace, load_user_profile


def test_scheduler_runs_each_due_task_once(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "scheduler-test")
    now = datetime(2026, 6, 26, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    calls: list[tuple[str, str]] = []

    def fake_run(profile_path, task_type, scheduled_for, run_id):
        calls.append((task_type, run_id))
        return {
            "status": "awaiting_approval" if task_type != "monthly_research" else "completed",
            "run_dir": str(tmp_path / run_id),
            "approval_required": task_type != "monthly_research",
            "approval_payload": {"reports": {}},
        }

    monkeypatch.setattr("fund_agent.scheduler.run_agent_workflow", fake_run)

    first = run_due_agent_tasks(profile_path, now)
    second = run_due_agent_tasks(profile_path, now)

    assert {result["task_type"] for result in first} == {
        "daily_rebalance",
        "weekly_screen",
        "monthly_research",
    }
    assert second == []
    assert len(calls) == 3
    records = load_scheduler_runs(load_user_profile(profile_path))
    assert len(records) == 3
    assert {record["status"] for record in records} == {"awaiting_approval", "completed"}


def test_scheduler_reports_no_task_before_research_time(tmp_path: Path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "scheduler-test")
    now = datetime(2026, 6, 26, 14, 29, tzinfo=ZoneInfo("Asia/Shanghai"))

    tasks, source = scheduled_tasks(profile_path, now)

    assert source == "weekday fallback configured by user"
    assert tasks == []
