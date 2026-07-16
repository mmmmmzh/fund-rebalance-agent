from __future__ import annotations

from pathlib import Path

import pandas as pd

from fund_agent.config import FundSpec


def load_fund_universe_csv(path: str | Path) -> tuple[FundSpec, ...]:
    csv_path = Path(path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    required = {"code", "name", "category", "is_equity_like"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Universe file {csv_path} missing columns: {sorted(missing)}")

    specs = []
    for row in df.to_dict(orient="records"):
        code = row["code"].strip()
        name = row["name"].strip()
        if not code or not name:
            continue
        specs.append(
            FundSpec(
                code=code,
                name=name,
                category=row["category"].strip() or "unknown",
                is_equity_like=_parse_bool(row["is_equity_like"]),
                instrument_type=(row.get("instrument_type") or "open_fund").strip(),
            )
        )
    if not specs:
        raise ValueError(f"Universe file {csv_path} has no usable funds.")
    return tuple(specs)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "equity", "权益"}:
        return True
    if normalized in {"0", "false", "no", "n", "defensive", "非权益"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")

