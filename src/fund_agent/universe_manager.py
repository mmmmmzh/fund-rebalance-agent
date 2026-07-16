from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fund_agent.user_profile import is_user_data_path


UNIVERSE_COLUMNS = [
    "code",
    "name",
    "category",
    "is_equity_like",
    "instrument_type",
    "status",
    "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a local research-universe CSV.")
    parser.add_argument("--file", required=True, help="CSV inside user_data/.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--code", required=True)
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument("--category", default="custom")
    add_parser.add_argument("--equity-like", choices=["true", "false"], default="true")
    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--code", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.file).expanduser().resolve()
    if not is_user_data_path(path):
        raise ValueError("Universe files must stay inside user_data/.")
    if args.command == "list":
        print(_read_universe(path).to_string(index=False))
    elif args.command == "add":
        add_fund(path, args.code, args.name, args.category, args.equity_like == "true")
    else:
        remove_fund(path, args.code)
    return 0


def add_fund(path: Path, code: str, name: str, category: str, is_equity_like: bool) -> None:
    frame = _read_universe(path)
    normalized = _normalize_code(code)
    frame = frame[frame["code"].astype(str).map(_normalize_code) != normalized]
    row = {
        "code": normalized,
        "name": str(name).strip(),
        "category": str(category).strip(),
        "is_equity_like": str(bool(is_equity_like)).lower(),
        "instrument_type": "research_fund",
        "status": "manual",
        "notes": "Local research entry",
    }
    _write_universe(path, pd.concat([frame, pd.DataFrame([row])], ignore_index=True))


def remove_fund(path: Path, code: str) -> bool:
    frame = _read_universe(path)
    normalized = _normalize_code(code)
    keep = frame["code"].astype(str).map(_normalize_code) != normalized
    removed = bool((~keep).any())
    _write_universe(path, frame.loc[keep].copy())
    return removed


def _read_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in UNIVERSE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[UNIVERSE_COLUMNS]


def _write_universe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[UNIVERSE_COLUMNS].to_csv(path, index=False, encoding="utf-8")


def _normalize_code(code: str) -> str:
    value = str(code).strip()
    if not value.isdigit() or len(value) > 12:
        raise ValueError("code must contain 1-12 digits")
    return value.zfill(6)


if __name__ == "__main__":
    raise SystemExit(main())
