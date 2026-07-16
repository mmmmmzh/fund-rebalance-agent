from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "signal_id",
    "run_id",
    "generated_at",
    "code",
    "name",
    "signal",
    "signal_score",
    "signal_confidence",
    "reference_nav_date",
    "reference_nav",
    "review_status",
]


def ledger_columns(horizons: Iterable[int] = (5, 20, 60)) -> list[str]:
    columns = list(BASE_COLUMNS)
    for horizon in sorted(set(int(value) for value in horizons)):
        columns.extend([f"fund_return_{horizon}d", f"signal_return_{horizon}d"])
    return columns


def load_paper_ledger(
    path: str | Path,
    horizons: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    target = Path(path)
    columns = ledger_columns(horizons)
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(target, dtype={"code": str})
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return frame.reindex(columns=columns)


def record_run_recommendations(
    ledger_path: str | Path,
    state: dict,
    horizons: Iterable[int] = (5, 20, 60),
) -> int:
    """Record research signals for forward validation.

    The legacy function name is retained for API compatibility. The public
    edition records no order, amount, fee or execution field.
    """

    if state.get("task_type") != "daily_rebalance":
        return 0
    signals = [
        item
        for item in state.get("research_actions", state.get("proposed_actions", []))
        if item.get("signal") in {"strong", "weak", "risk_exit"}
    ]
    if not signals:
        return 0
    normalized_horizons = sorted(set(int(value) for value in horizons))
    path = Path(ledger_path)
    ledger = load_paper_ledger(path, normalized_horizons)
    existing = set(ledger["signal_id"].dropna().astype(str))
    prices = _load_prices(state.get("prices_path"))
    rows = []
    for signal in signals:
        code = str(signal["code"]).zfill(6)
        signal_id = _signal_id(str(state.get("run_id", "")), code, str(signal["signal"]))
        if signal_id in existing:
            continue
        nav_date, nav = _reference_nav(prices, code)
        rows.append(
            {
                "signal_id": signal_id,
                "run_id": state.get("run_id"),
                "generated_at": state.get("scheduled_for"),
                "code": code,
                "name": signal.get("name", code),
                "signal": signal["signal"],
                "signal_score": signal.get("signal_score"),
                "signal_confidence": signal.get("signal_confidence"),
                "reference_nav_date": nav_date,
                "reference_nav": nav,
                "review_status": state.get("status", "unreviewed"),
            }
        )
    if not rows:
        return 0
    ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    _write_ledger(ledger, path, normalized_horizons)
    return len(rows)


def refresh_recommendation_outcomes(
    ledger_path: str | Path,
    prices: pd.DataFrame,
    horizons: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    normalized_horizons = sorted(set(int(value) for value in horizons))
    path = Path(ledger_path)
    ledger = load_paper_ledger(path, normalized_horizons)
    if ledger.empty:
        return ledger
    clean_prices = prices.copy().sort_index()
    clean_prices.index = pd.to_datetime(clean_prices.index)
    clean_prices.columns = [str(column).zfill(6) for column in clean_prices.columns]
    clean_prices = clean_prices.apply(pd.to_numeric, errors="coerce")
    for index, row in ledger.iterrows():
        code = str(row["code"]).zfill(6)
        if code not in clean_prices:
            continue
        series = clean_prices[code].dropna()
        reference_date = pd.to_datetime(row.get("reference_nav_date"), errors="coerce")
        if pd.isna(reference_date):
            continue
        positions = np.flatnonzero(series.index >= reference_date)
        if len(positions) == 0:
            continue
        start_pos = int(positions[0])
        start_nav = (
            float(row.get("reference_nav"))
            if pd.notna(row.get("reference_nav"))
            else float(series.iloc[start_pos])
        )
        direction = 1.0 if row.get("signal") == "strong" else -1.0
        for horizon in normalized_horizons:
            target_pos = start_pos + horizon
            if target_pos >= len(series) or start_nav <= 0:
                continue
            fund_return = float(series.iloc[target_pos] / start_nav - 1.0)
            ledger.loc[index, f"fund_return_{horizon}d"] = fund_return
            ledger.loc[index, f"signal_return_{horizon}d"] = direction * fund_return
    _write_ledger(ledger, path, normalized_horizons)
    return ledger


def _signal_id(run_id: str, code: str, signal: str) -> str:
    return sha256(f"{run_id}|{code}|{signal}".encode()).hexdigest()[:20]


def _load_prices(value: object) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()
    path = Path(str(value))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _reference_nav(prices: pd.DataFrame, code: str) -> tuple[str | None, float | None]:
    if prices.empty or code not in prices:
        return None, None
    series = pd.to_numeric(prices[code], errors="coerce").dropna()
    if series.empty:
        return None, None
    return str(pd.Timestamp(series.index[-1]).date()), float(series.iloc[-1])


def _write_ledger(frame: pd.DataFrame, path: Path, horizons: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=ledger_columns(horizons)).to_csv(path, index=False, encoding="utf-8")
