from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.config import FundSpec


CONTEXT_COLUMNS = [
    "intraday_change",
    "intraday_confidence",
    "sector_change",
    "sector_confidence",
    "news_sentiment",
    "news_confidence",
    "policy_sentiment",
    "policy_confidence",
    "news_count",
    "news_evidence",
]


@dataclass(frozen=True)
class MarketContextSnapshot:
    signals: pd.DataFrame
    source: str
    fetched_at: pd.Timestamp
    market_data_as_of: str | None
    notes: tuple[str, ...] = ()


def load_market_context(
    source: str,
    fund_universe: tuple[FundSpec, ...],
    prices: pd.DataFrame,
    **_: object,
) -> MarketContextSnapshot | None:
    normalized = source.strip().lower()
    if normalized in {"none", "off"}:
        return None
    if normalized == "sample":
        return load_sample_market_context(fund_universe, prices)
    if normalized == "plugin":
        loader = require_adapter("market_context", get_adapters().market_context)
        snapshot = loader(fund_universe, prices)
        if not isinstance(snapshot, MarketContextSnapshot):
            raise TypeError("Market context adapter must return MarketContextSnapshot.")
        return _validated_snapshot(snapshot, fund_universe)
    raise ValueError(f"Unsupported market context source: {source}")


def load_sample_market_context(
    fund_universe: tuple[FundSpec, ...],
    prices: pd.DataFrame,
) -> MarketContextSnapshot:
    returns = prices.pct_change(fill_method=None)
    rows = []
    for index, fund in enumerate(fund_universe):
        series = returns.get(fund.code, pd.Series(dtype=float)).dropna()
        recent = float(series.tail(5).mean()) if not series.empty else np.nan
        sector = float(series.tail(20).mean()) if not series.empty else np.nan
        rows.append(
            {
                "code": fund.code,
                "intraday_change": recent,
                "intraday_confidence": 0.35,
                "sector_change": sector,
                "sector_confidence": 0.35,
                "news_sentiment": np.nan,
                "news_confidence": 0.0,
                "policy_sentiment": np.nan,
                "policy_confidence": 0.0,
                "news_count": 0,
                "news_evidence": "",
            }
        )
    signals = pd.DataFrame(rows).set_index("code").reindex(columns=CONTEXT_COLUMNS)
    return MarketContextSnapshot(
        signals=signals,
        source="deterministic synthetic context",
        fetched_at=pd.Timestamp.now(tz="Asia/Shanghai"),
        market_data_as_of=str(pd.Timestamp(prices.index[-1]).date()),
        notes=("Synthetic context is for workflow demonstration only.",),
    )


def _validated_snapshot(
    snapshot: MarketContextSnapshot,
    universe: tuple[FundSpec, ...],
) -> MarketContextSnapshot:
    signals = snapshot.signals.copy()
    signals.index = signals.index.astype(str).str.zfill(6)
    for column in CONTEXT_COLUMNS:
        if column not in signals:
            signals[column] = "" if column == "news_evidence" else np.nan
    codes = [fund.code for fund in universe]
    signals = signals.reindex(codes)[CONTEXT_COLUMNS]
    return MarketContextSnapshot(
        signals=signals,
        source=str(snapshot.source)[:200],
        fetched_at=pd.Timestamp(snapshot.fetched_at),
        market_data_as_of=snapshot.market_data_as_of,
        notes=tuple(str(note)[:500] for note in snapshot.notes),
    )
