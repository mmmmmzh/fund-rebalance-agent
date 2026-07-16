from __future__ import annotations

import argparse
from datetime import datetime
import json

from fund_agent.agent_workflow import (
    resume_agent_workflow,
    run_agent_workflow,
    workflow_status,
)


DEFAULT_PROFILE = "config/demo_profile.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and review the LangGraph fund research agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "task",
        choices=["daily_rebalance", "weekly_screen", "monthly_research"],
    )
    run_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--scheduled-for", default=None, help="ISO datetime for reproducible runs.")
    for command in ["approve", "reject"]:
        review_parser = subparsers.add_parser(command)
        review_parser.add_argument("--profile", default=DEFAULT_PROFILE)
        review_parser.add_argument("--run-id", required=True)
        review_parser.add_argument("--feedback", default="")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    status_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        scheduled_for = datetime.fromisoformat(args.scheduled_for) if args.scheduled_for else None
        result = run_agent_workflow(
            args.profile,
            args.task,
            scheduled_for=scheduled_for,
            run_id=args.run_id,
        )
    elif args.command in {"approve", "reject"}:
        result = resume_agent_workflow(
            args.profile,
            args.run_id,
            decision=args.command,
            feedback=args.feedback,
        )
    else:
        result = workflow_status(args.profile, args.run_id)
    _print_result(result)
    return 0


def _print_result(result: dict) -> None:
    print(f"run_id={result.get('run_id', '')}")
    print(f"task_type={result.get('task_type', '')}")
    print(f"status={result.get('status', '')}")
    if result.get("interrupts"):
        print("approval_required=true")
        print(json.dumps(result["interrupts"], ensure_ascii=False, indent=2))
    for event in result.get("trace", []):
        print(f"agent={event.get('agent')}: {event.get('summary')}")
    for key in ["analysis_report_path", "candidate_report_path", "monthly_report_path"]:
        if result.get(key):
            print(f"{key}={result[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
