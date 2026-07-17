from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from fund_agent.agent_workflow import run_agent_workflow, workflow_status
from fund_agent.schedule import ScheduledTask, due_tasks, load_trade_dates
from fund_agent.user_profile import LoadedProfile, load_user_profile


TERMINAL_STATUSES = {
    "completed",
    "awaiting_approval",
    "approved",
    "rejected",
    "blocked_by_risk",
}
LEASE_DURATION = timedelta(minutes=30)


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
        if record.get("status") in TERMINAL_STATUSES
    }
    return due_tasks(localized, loaded.profile, trade_dates, completed), calendar_source


def run_due_agent_tasks(
    profile_path: str | Path,
    now: datetime | None = None,
    refresh_calendar: bool = False,
) -> list[dict[str, Any]]:
    loaded = load_user_profile(profile_path)
    tasks, calendar_source = scheduled_tasks(loaded.path, now, refresh_calendar)
    results: list[dict[str, Any]] = []
    for task in tasks:
        claim = _claim_scheduled_task(loaded, task, calendar_source)
        if claim is None:
            continue
        try:
            result = run_agent_workflow(
                loaded.path,
                task.task_type,
                scheduled_for=task.scheduled_at,
                run_id=claim["run_id"],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            finished = _finish_claim(loaded, claim, "failed", error=error)
            results.append(finished or {**claim, "status": "superseded", "error": error})
            continue

        status = str(result.get("status", "unknown"))
        error = None
        if status not in TERMINAL_STATUSES:
            error = f"Workflow returned non-terminal status: {status}"
            status = "failed"
        finished = _finish_claim(
            loaded,
            claim,
            status,
            error=error,
            run_dir=result.get("run_dir"),
            approval_required=bool(result.get("approval_required", False)),
            reports=result.get("approval_payload", {}).get("reports", {}),
        )
        results.append(finished or {**claim, "status": "superseded", "error": error})
    return results


def load_scheduler_runs(loaded: LoadedProfile) -> list[dict[str, Any]]:
    with closing(_connect(loaded)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM scheduler_runs
            ORDER BY scheduled_date DESC, task_type ASC
            """
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = _row_to_record(row)
            attempts = connection.execute(
                """
                SELECT attempt, run_id, status, started_at, lease_expires_at,
                       finished_at, error
                FROM scheduler_attempts
                WHERE task_type = ? AND scheduled_date = ?
                ORDER BY attempt ASC
                """,
                (record["task_type"], record["scheduled_date"]),
            ).fetchall()
            record["attempt_history"] = [dict(attempt) for attempt in attempts]
            records.append(record)
    return records


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


def _claim_scheduled_task(
    loaded: LoadedProfile,
    task: ScheduledTask,
    calendar_source: str,
    claimed_at: datetime | None = None,
) -> dict[str, Any] | None:
    now = _localized_now(loaded, claimed_at)
    scheduled_date = task.scheduled_at.date().isoformat()
    observed = _load_run(loaded, task.task_type, scheduled_date)
    recovered: dict[str, Any] | None = None
    if observed and observed["status"] == "starting" and _lease_expired(observed, now):
        recovered = _recover_terminal_state(loaded, observed)

    with closing(_connect(loaded)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT * FROM scheduler_runs
                WHERE task_type = ? AND scheduled_date = ?
                """,
                (task.task_type, scheduled_date),
            ).fetchone()
            current = _row_to_record(row) if row is not None else None
            if current and current["status"] in TERMINAL_STATUSES:
                connection.commit()
                return None
            if current and current["status"] == "starting":
                if not _lease_expired(current, now):
                    connection.commit()
                    return None
                if (
                    recovered is not None
                    and observed is not None
                    and current["run_id"] == observed["run_id"]
                ):
                    _repair_terminal_run(connection, current, recovered, now)
                    connection.commit()
                    return None
                _abandon_attempt(connection, current, now)

            attempt = int(current["attempt"]) + 1 if current else 1
            run_id = (
                f"{task.task_type}-{task.scheduled_at.strftime('%Y%m%d-%H%M')}"
                f"-a{attempt}-{uuid4().hex[:6]}"
            )
            started_at = now.isoformat()
            lease_expires_at = (now + LEASE_DURATION).isoformat()
            run_dir = _run_dir(loaded, run_id)
            connection.execute(
                """
                INSERT INTO scheduler_attempts (
                    task_type, scheduled_date, attempt, run_id, status,
                    started_at, lease_expires_at
                ) VALUES (?, ?, ?, ?, 'starting', ?, ?)
                """,
                (
                    task.task_type,
                    scheduled_date,
                    attempt,
                    run_id,
                    started_at,
                    lease_expires_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO scheduler_runs (
                    task_type, scheduled_date, scheduled_at, run_id, status,
                    calendar_source, started_at, lease_expires_at, run_dir,
                    attempt, approval_required, reports_json, updated_at
                ) VALUES (?, ?, ?, ?, 'starting', ?, ?, ?, ?, ?, 0, '{}', ?)
                ON CONFLICT(task_type, scheduled_date) DO UPDATE SET
                    scheduled_at = excluded.scheduled_at,
                    run_id = excluded.run_id,
                    status = excluded.status,
                    calendar_source = excluded.calendar_source,
                    started_at = excluded.started_at,
                    lease_expires_at = excluded.lease_expires_at,
                    finished_at = NULL,
                    run_dir = excluded.run_dir,
                    error = NULL,
                    attempt = excluded.attempt,
                    approval_required = 0,
                    reports_json = '{}',
                    updated_at = excluded.updated_at
                """,
                (
                    task.task_type,
                    scheduled_date,
                    task.scheduled_at.isoformat(),
                    run_id,
                    calendar_source,
                    started_at,
                    lease_expires_at,
                    str(run_dir),
                    attempt,
                    started_at,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return _load_run(loaded, task.task_type, scheduled_date)


def _finish_claim(
    loaded: LoadedProfile,
    claim: dict[str, Any],
    status: str,
    *,
    error: str | None = None,
    run_dir: str | None = None,
    approval_required: bool = False,
    reports: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    now = datetime.now(ZoneInfo(loaded.profile.timezone)).isoformat()
    reports_json = json.dumps(reports or {}, ensure_ascii=False)
    with closing(_connect(loaded)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                """
                UPDATE scheduler_runs
                SET status = ?, finished_at = ?, lease_expires_at = NULL,
                    run_dir = COALESCE(?, run_dir), error = ?,
                    approval_required = ?, reports_json = ?, updated_at = ?
                WHERE task_type = ? AND scheduled_date = ? AND run_id = ?
                      AND attempt = ? AND status = 'starting'
                """,
                (
                    status,
                    now,
                    run_dir,
                    error,
                    int(approval_required),
                    reports_json,
                    now,
                    claim["task_type"],
                    claim["scheduled_date"],
                    claim["run_id"],
                    claim["attempt"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE scheduler_attempts
                SET status = ?, finished_at = ?, lease_expires_at = NULL, error = ?
                WHERE task_type = ? AND scheduled_date = ? AND attempt = ?
                      AND run_id = ? AND status = 'starting'
                """,
                (
                    status,
                    now,
                    error,
                    claim["task_type"],
                    claim["scheduled_date"],
                    claim["attempt"],
                    claim["run_id"],
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return _load_run(loaded, claim["task_type"], claim["scheduled_date"])


def _recover_terminal_state(
    loaded: LoadedProfile,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    try:
        candidates.append(workflow_status(loaded.path, record["run_id"]))
    except Exception:
        pass
    run_state_path = Path(record.get("run_dir") or _run_dir(loaded, record["run_id"])) / "run_state.json"
    if run_state_path.exists():
        try:
            payload = json.loads(run_state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                candidates.append(payload)
        except (OSError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        if candidate.get("status") in TERMINAL_STATUSES:
            return candidate
    return None


def _repair_terminal_run(
    connection: sqlite3.Connection,
    current: dict[str, Any],
    recovered: dict[str, Any],
    now: datetime,
) -> None:
    status = str(recovered["status"])
    finished_at = now.isoformat()
    run_dir = recovered.get("run_dir") or current.get("run_dir")
    reports = recovered.get("approval_payload", {}).get("reports", {})
    connection.execute(
        """
        UPDATE scheduler_runs
        SET status = ?, finished_at = ?, lease_expires_at = NULL,
            run_dir = ?, approval_required = ?, reports_json = ?, updated_at = ?
        WHERE task_type = ? AND scheduled_date = ? AND run_id = ? AND attempt = ?
        """,
        (
            status,
            finished_at,
            run_dir,
            int(bool(recovered.get("approval_required", False))),
            json.dumps(reports, ensure_ascii=False),
            finished_at,
            current["task_type"],
            current["scheduled_date"],
            current["run_id"],
            current["attempt"],
        ),
    )
    connection.execute(
        """
        UPDATE scheduler_attempts
        SET status = ?, finished_at = ?, lease_expires_at = NULL
        WHERE task_type = ? AND scheduled_date = ? AND attempt = ? AND run_id = ?
        """,
        (
            status,
            finished_at,
            current["task_type"],
            current["scheduled_date"],
            current["attempt"],
            current["run_id"],
        ),
    )


def _abandon_attempt(
    connection: sqlite3.Connection,
    current: dict[str, Any],
    now: datetime,
) -> None:
    message = "Lease expired without a terminal workflow state; attempt abandoned."
    if current.get("error"):
        message = f"{current['error']} | {message}"
    finished_at = now.isoformat()
    connection.execute(
        """
        UPDATE scheduler_attempts
        SET status = 'abandoned', finished_at = ?, lease_expires_at = NULL, error = ?
        WHERE task_type = ? AND scheduled_date = ? AND attempt = ? AND run_id = ?
        """,
        (
            finished_at,
            message,
            current["task_type"],
            current["scheduled_date"],
            current["attempt"],
            current["run_id"],
        ),
    )


def _load_run(
    loaded: LoadedProfile,
    task_type: str,
    scheduled_date: str,
) -> dict[str, Any] | None:
    with closing(_connect(loaded)) as connection:
        row = connection.execute(
            """
            SELECT * FROM scheduler_runs
            WHERE task_type = ? AND scheduled_date = ?
            """,
            (task_type, scheduled_date),
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["approval_required"] = bool(record.get("approval_required", 0))
    try:
        record["reports"] = json.loads(record.pop("reports_json", "{}") or "{}")
    except json.JSONDecodeError:
        record["reports"] = {}
    return record


def _lease_expired(record: dict[str, Any], now: datetime) -> bool:
    value = record.get("lease_expires_at")
    if not value:
        return True
    try:
        expires_at = datetime.fromisoformat(str(value))
    except ValueError:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=now.tzinfo)
    return expires_at <= now


def _connect(loaded: LoadedProfile) -> sqlite3.Connection:
    path = _registry_path(loaded)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scheduler_runs (
            task_type TEXT NOT NULL,
            scheduled_date TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,
            run_id TEXT NOT NULL,
            status TEXT NOT NULL,
            calendar_source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            lease_expires_at TEXT,
            finished_at TEXT,
            run_dir TEXT,
            error TEXT,
            attempt INTEGER NOT NULL CHECK (attempt >= 1),
            approval_required INTEGER NOT NULL DEFAULT 0,
            reports_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (task_type, scheduled_date)
        );
        CREATE TABLE IF NOT EXISTS scheduler_attempts (
            task_type TEXT NOT NULL,
            scheduled_date TEXT NOT NULL,
            attempt INTEGER NOT NULL CHECK (attempt >= 1),
            run_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            lease_expires_at TEXT,
            finished_at TEXT,
            error TEXT,
            PRIMARY KEY (task_type, scheduled_date, attempt)
        );
        CREATE TABLE IF NOT EXISTS scheduler_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    _migrate_legacy_registry(connection, loaded)
    return connection


def _migrate_legacy_registry(
    connection: sqlite3.Connection,
    loaded: LoadedProfile,
) -> None:
    marker = connection.execute(
        "SELECT value FROM scheduler_meta WHERE key = 'legacy_json_migrated'"
    ).fetchone()
    if marker is not None:
        return
    legacy_path = _legacy_registry_path(loaded)
    records: list[dict[str, Any]] = []
    if legacy_path.exists():
        payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        raw_records = payload.get("runs", [])
        if not isinstance(raw_records, list):
            raise ValueError(f"Invalid scheduler registry: {legacy_path}")
        records = [dict(record) for record in raw_records]

    connection.execute("BEGIN IMMEDIATE")
    try:
        marker = connection.execute(
            "SELECT value FROM scheduler_meta WHERE key = 'legacy_json_migrated'"
        ).fetchone()
        if marker is None:
            for index, record in enumerate(records, start=1):
                _insert_legacy_record(connection, record, index)
            connection.execute(
                "INSERT INTO scheduler_meta(key, value) VALUES ('legacy_json_migrated', '1')"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _insert_legacy_record(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    fallback_attempt: int,
) -> None:
    task_type = str(record.get("task_type", "unknown"))
    scheduled_date = str(record.get("scheduled_date", ""))
    if not scheduled_date:
        return
    attempt = int(record.get("attempt", fallback_attempt))
    run_id = str(record.get("run_id", f"legacy-{task_type}-{scheduled_date}-{attempt}"))
    status = str(record.get("status", "failed"))
    started_at = str(record.get("started_at", record.get("scheduled_at", datetime.now().isoformat())))
    lease_expires_at = record.get("lease_expires_at")
    if status == "starting" and not lease_expires_at:
        lease_expires_at = started_at
    finished_at = record.get("finished_at")
    connection.execute(
        """
        INSERT OR REPLACE INTO scheduler_attempts (
            task_type, scheduled_date, attempt, run_id, status, started_at,
            lease_expires_at, finished_at, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_type,
            scheduled_date,
            attempt,
            run_id,
            status,
            started_at,
            lease_expires_at,
            finished_at,
            record.get("error"),
        ),
    )
    connection.execute(
        """
        INSERT INTO scheduler_runs (
            task_type, scheduled_date, scheduled_at, run_id, status,
            calendar_source, started_at, lease_expires_at, finished_at,
            run_dir, error, attempt, approval_required, reports_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_type, scheduled_date) DO UPDATE SET
            scheduled_at = excluded.scheduled_at,
            run_id = excluded.run_id,
            status = excluded.status,
            calendar_source = excluded.calendar_source,
            started_at = excluded.started_at,
            lease_expires_at = excluded.lease_expires_at,
            finished_at = excluded.finished_at,
            run_dir = excluded.run_dir,
            error = excluded.error,
            attempt = excluded.attempt,
            approval_required = excluded.approval_required,
            reports_json = excluded.reports_json,
            updated_at = excluded.updated_at
        """,
        (
            task_type,
            scheduled_date,
            str(record.get("scheduled_at", f"{scheduled_date}T00:00:00")),
            run_id,
            status,
            str(record.get("calendar_source", "legacy-json")),
            started_at,
            lease_expires_at,
            finished_at,
            record.get("run_dir"),
            record.get("error"),
            attempt,
            int(bool(record.get("approval_required", False))),
            json.dumps(record.get("reports", {}), ensure_ascii=False),
            str(finished_at or started_at),
        ),
    )


def _registry_path(loaded: LoadedProfile) -> Path:
    state_dir = loaded.resolve(loaded.profile.state_dir)
    if state_dir is None:
        raise ValueError("state_dir is required")
    return state_dir / "scheduler_runs.sqlite3"


def _legacy_registry_path(loaded: LoadedProfile) -> Path:
    return _registry_path(loaded).with_name("scheduler_runs.json")


def _run_dir(loaded: LoadedProfile, run_id: str) -> Path:
    output_root = loaded.resolve(loaded.profile.output_root)
    if output_root is None:
        raise ValueError("output_root is required")
    return (output_root / run_id).resolve()


def _localized_now(loaded: LoadedProfile, now: datetime | None) -> datetime:
    zone = ZoneInfo(loaded.profile.timezone)
    if now is None:
        return datetime.now(zone)
    return now.astimezone(zone) if now.tzinfo else now.replace(tzinfo=zone)


if __name__ == "__main__":
    raise SystemExit(main())
