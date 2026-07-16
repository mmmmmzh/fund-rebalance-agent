from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from fund_agent.agent_workflow import run_agent_workflow
from fund_agent.schedule import ScheduledTask, due_tasks, load_trade_dates
from fund_agent.user_profile import LoadedProfile, load_user_profile


def scheduled_tasks(
    profile_path: str | Path,
    now: datetime | None = None,
    refresh_calendar: bool = False,
) -> tuple[list[ScheduledTask], str]:
    loaded = load_user_profile(profile_path)
    localized = _localized_now(loaded, now)
    trade_dates, calendar_source = load_trade_dates(loaded, refresh=refresh_calendar)
    completed = {
        (str(record["task_type"]), date.fromisoformat(str(record["scheduled_date"])))
        for record in load_scheduler_runs(loaded)
        if record.get("status") != "failed"
    }
    return due_tasks(localized, loaded.profile, trade_dates, completed), calendar_source


def run_due_agent_tasks(
    profile_path: str | Path,
    now: datetime | None = None,
    refresh_calendar: bool = False,
) -> list[dict[str, Any]]:
    loaded = load_user_profile(profile_path)
    tasks, calendar_source = scheduled_tasks(loaded.path, now, refresh_calendar)
    records = load_scheduler_runs(loaded)
    results: list[dict[str, Any]] = []
    for task in tasks:
        run_id = (
            f"{task.task_type}-{task.scheduled_at.strftime('%Y%m%d-%H%M')}"
            f"-{uuid4().hex[:6]}"
        )
        record = {
            "task_type": task.task_type,
            "scheduled_date": task.scheduled_at.date().isoformat(),
            "scheduled_at": task.scheduled_at.isoformat(),
            "run_id": run_id,
            "status": "starting",
            "calendar_source": calendar_source,
            "started_at": datetime.now(ZoneInfo(loaded.profile.timezone)).isoformat(),
        }
        records.append(record)
        save_scheduler_runs(loaded, records)
        try:
            result = run_agent_workflow(
                loaded.path,
                task.task_type,
                scheduled_for=task.scheduled_at,
                run_id=run_id,
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["finished_at"] = datetime.now(
                ZoneInfo(loaded.profile.timezone)
            ).isoformat()
            save_scheduler_runs(loaded, records)
            results.append(dict(record))
            continue
        record["status"] = result.get("status", "unknown")
        record["run_dir"] = result.get("run_dir")
        record["finished_at"] = datetime.now(ZoneInfo(loaded.profile.timezone)).isoformat()
        save_scheduler_runs(loaded, records)
        results.append(
            {
                **record,
                "approval_required": result.get("approval_required", False),
                "reports": result.get("approval_payload", {}).get("reports", {}),
            }
        )
    return results


def load_scheduler_runs(loaded: LoadedProfile) -> list[dict[str, Any]]:
    path = _registry_path(loaded)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"Invalid scheduler registry: {path}")
    return [dict(record) for record in runs]


def save_scheduler_runs(loaded: LoadedProfile, records: list[dict[str, Any]]) -> None:
    path = _registry_path(loaded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"runs": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run due fund research Agent tasks once.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("due", "run-due"):
        child = subparsers.add_parser(name)
        child.add_argument("--profile", required=True)
        child.add_argument("--at", help="ISO datetime; defaults to current profile timezone.")
        child.add_argument("--refresh-calendar", action="store_true")
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--profile", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = load_user_profile(args.profile)
    if args.command == "list":
        print(json.dumps(load_scheduler_runs(loaded), ensure_ascii=False, indent=2))
        return 0
    now = datetime.fromisoformat(args.at) if args.at else None
    if args.command == "due":
        tasks, source = scheduled_tasks(loaded.path, now, args.refresh_calendar)
        print(
            json.dumps(
                {
                    "calendar_source": source,
                    "tasks": [
                        {
                            "task_type": task.task_type,
                            "scheduled_at": task.scheduled_at.isoformat(),
                            "reason": task.reason,
                        }
                        for task in tasks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    results = run_due_agent_tasks(loaded.path, now, args.refresh_calendar)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(result.get("status") == "failed" for result in results) else 0


def _registry_path(loaded: LoadedProfile) -> Path:
    state_dir = loaded.resolve(loaded.profile.state_dir)
    if state_dir is None:
        raise ValueError("state_dir is required")
    return state_dir / "scheduler_runs.json"


def _localized_now(loaded: LoadedProfile, now: datetime | None) -> datetime:
    zone = ZoneInfo(loaded.profile.timezone)
    if now is None:
        return datetime.now(zone)
    return now.astimezone(zone) if now.tzinfo else now.replace(tzinfo=zone)


if __name__ == "__main__":
    raise SystemExit(main())
