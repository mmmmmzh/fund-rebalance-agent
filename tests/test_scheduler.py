from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Barrier, Lock
import time
from zoneinfo import ZoneInfo

from fund_agent.scheduler import (
    LEASE_DURATION,
    _claim_scheduled_task,
    load_scheduler_runs,
    run_due_agent_tasks,
    scheduled_tasks,
)
from fund_agent.user_profile import initialize_user_workspace, load_user_profile


ZONE = ZoneInfo("Asia/Shanghai")
DAILY_NOW = datetime(2026, 6, 29, 14, 31, tzinfo=ZONE)


def _completed_result(tmp_path: Path, run_id: str) -> dict[str, object]:
    return {
        "status": "completed",
        "run_dir": str(tmp_path / run_id),
        "approval_required": False,
        "approval_payload": {"reports": {}},
    }


def test_scheduler_runs_each_due_task_once(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "scheduler-test")
    now = datetime(2026, 6, 26, 20, 0, tzinfo=ZONE)
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
    now = datetime(2026, 6, 26, 14, 29, tzinfo=ZONE)

    tasks, source = scheduled_tasks(profile_path, now)

    assert source == "weekday fallback configured by user"
    assert tasks == []


def test_fresh_starting_lease_is_not_executed_again(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "fresh-lease")
    loaded = load_user_profile(profile_path)
    tasks, source = scheduled_tasks(profile_path, DAILY_NOW)
    claim = _claim_scheduled_task(loaded, tasks[0], source)
    calls: list[str] = []

    monkeypatch.setattr(
        "fund_agent.scheduler.run_agent_workflow",
        lambda *args, **kwargs: calls.append("unexpected"),
    )

    assert run_due_agent_tasks(profile_path, DAILY_NOW) == []
    assert calls == []
    record = load_scheduler_runs(loaded)[0]
    assert record["status"] == "starting"
    assert record["run_id"] == claim["run_id"]
    assert record["attempt"] == 1


def test_expired_starting_repairs_terminal_workflow_state(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "recover-terminal")
    loaded = load_user_profile(profile_path)
    tasks, source = scheduled_tasks(profile_path, DAILY_NOW)
    expired_at = datetime.now(ZONE) - LEASE_DURATION - timedelta(minutes=1)
    claim = _claim_scheduled_task(loaded, tasks[0], source, claimed_at=expired_at)

    monkeypatch.setattr(
        "fund_agent.scheduler.workflow_status",
        lambda profile_path, run_id: {
            **_completed_result(tmp_path, run_id),
            "run_id": run_id,
        },
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("terminal workflow must not be executed again")

    monkeypatch.setattr("fund_agent.scheduler.run_agent_workflow", unexpected_run)

    assert run_due_agent_tasks(profile_path, DAILY_NOW) == []
    record = load_scheduler_runs(loaded)[0]
    assert record["status"] == "completed"
    assert record["run_id"] == claim["run_id"]
    assert record["attempt"] == 1
    assert record["attempt_history"][0]["status"] == "completed"


def test_expired_starting_is_abandoned_before_retry(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "recover-abandoned")
    loaded = load_user_profile(profile_path)
    tasks, source = scheduled_tasks(profile_path, DAILY_NOW)
    expired_at = datetime.now(ZONE) - LEASE_DURATION - timedelta(minutes=1)
    old_claim = _claim_scheduled_task(loaded, tasks[0], source, claimed_at=expired_at)

    def missing_workflow_state(*args, **kwargs):
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr("fund_agent.scheduler.workflow_status", missing_workflow_state)
    monkeypatch.setattr(
        "fund_agent.scheduler.run_agent_workflow",
        lambda profile_path, task_type, scheduled_for, run_id: _completed_result(
            tmp_path, run_id
        ),
    )

    result = run_due_agent_tasks(profile_path, DAILY_NOW)

    assert len(result) == 1
    assert result[0]["status"] == "completed"
    record = load_scheduler_runs(loaded)[0]
    assert record["attempt"] == 2
    assert record["run_id"] != old_claim["run_id"]
    assert [attempt["status"] for attempt in record["attempt_history"]] == [
        "abandoned",
        "completed",
    ]
    assert record["attempt_history"][0]["run_id"] == old_claim["run_id"]
    assert "Lease expired" in record["attempt_history"][0]["error"]


def test_failed_attempt_can_retry_with_auditable_history(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "retry-failure")
    calls = 0

    def flaky_run(profile_path, task_type, scheduled_for, run_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return _completed_result(tmp_path, run_id)

    monkeypatch.setattr("fund_agent.scheduler.run_agent_workflow", flaky_run)

    first = run_due_agent_tasks(profile_path, DAILY_NOW)
    second = run_due_agent_tasks(profile_path, DAILY_NOW)

    assert first[0]["status"] == "failed"
    assert second[0]["status"] == "completed"
    record = load_scheduler_runs(load_user_profile(profile_path))[0]
    assert record["attempt"] == 2
    assert [attempt["status"] for attempt in record["attempt_history"]] == [
        "failed",
        "completed",
    ]
    assert "temporary failure" in record["attempt_history"][0]["error"]


def test_concurrent_schedulers_claim_workflow_once(tmp_path: Path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "concurrent")
    barrier = Barrier(2)
    lock = Lock()
    calls: list[str] = []

    def slow_run(profile_path, task_type, scheduled_for, run_id):
        with lock:
            calls.append(run_id)
        time.sleep(0.2)
        return _completed_result(tmp_path, run_id)

    monkeypatch.setattr("fund_agent.scheduler.run_agent_workflow", slow_run)

    def invoke() -> list[dict[str, object]]:
        barrier.wait()
        return run_due_agent_tasks(profile_path, DAILY_NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: invoke(), range(2)))

    assert len(calls) == 1
    assert sum(len(output) for output in outputs) == 1
    assert load_scheduler_runs(load_user_profile(profile_path))[0]["status"] == "completed"


def test_legacy_json_registry_is_migrated_with_attempt_history(tmp_path: Path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "user", "legacy")
    loaded = load_user_profile(profile_path)
    state_dir = loaded.resolve(loaded.profile.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = state_dir / "scheduler_runs.json"
    legacy_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "task_type": "daily_rebalance",
                        "scheduled_date": "2026-06-29",
                        "scheduled_at": DAILY_NOW.isoformat(),
                        "run_id": "legacy-run-id",
                        "status": "failed",
                        "started_at": DAILY_NOW.isoformat(),
                        "finished_at": DAILY_NOW.isoformat(),
                        "error": "legacy failure",
                        "attempt": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    record = load_scheduler_runs(loaded)[0]

    assert record["run_id"] == "legacy-run-id"
    assert record["attempt"] == 3
    assert record["error"] == "legacy failure"
    assert record["attempt_history"] == [
        {
            "attempt": 3,
            "run_id": "legacy-run-id",
            "status": "failed",
            "started_at": DAILY_NOW.isoformat(),
            "lease_expires_at": None,
            "finished_at": DAILY_NOW.isoformat(),
            "error": "legacy failure",
        }
    ]
