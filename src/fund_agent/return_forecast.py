from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fund_agent.config import FundSpec
from fund_agent.market_context import MarketContextSnapshot
from fund_agent.metrics import compute_returns


CONTEXT_WEIGHTS = {
    "intraday_score": 0.35,
    "sector_score": 0.30,
    "news_score": 0.25,
    "policy_score": 0.10,
}


@dataclass(frozen=True)
class ExpectedReturnForecast:
    table: pd.DataFrame
    context_source: str
    context_fetched_at: pd.Timestamp
    context_market_data_as_of: str | None
    notes: tuple[str, ...]

    @property
    def expected_returns(self) -> pd.Series:
        return self.table["blended_expected_return"]


def build_expected_return_forecast(
    prices: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    context: MarketContextSnapshot,
    max_context_annual_tilt: float = 0.08,
    lookback_days: int | None = None,
    historical_return_bounds: tuple[float, float] = (-0.30, 0.50),
) -> ExpectedReturnForecast:
    returns = compute_returns(prices)
    if lookback_days is not None:
        returns = returns.tail(lookback_days)
    arithmetic_mean = returns.mean() * 252
    ewm_span = min(63, max(10, len(returns)))
    ewm_mean = returns.ewm(span=ewm_span, adjust=False).mean().iloc[-1] * 252
    raw_historical = 0.50 * arithmetic_mean + 0.50 * ewm_mean
    common_mean = float(raw_historical.median())
    historical = (0.65 * raw_historical + 0.35 * common_mean).clip(*historical_return_bounds)

    codes = [fund.code for fund in fund_universe if fund.code in prices.columns]
    signals = context.signals.reindex(codes)
    table = pd.DataFrame(index=codes)
    table["historical_annual_return"] = historical.reindex(codes)
    table["intraday_change"] = pd.to_numeric(signals["intraday_change"], errors="coerce")
    table["intraday_confidence"] = _component_confidence(signals, "intraday_confidence", "intraday_change")
    table["sector_change"] = pd.to_numeric(signals["sector_change"], errors="coerce")
    table["sector_confidence"] = _component_confidence(signals, "sector_confidence", "sector_change")
    table["news_sentiment"] = pd.to_numeric(signals["news_sentiment"], errors="coerce")
    table["news_confidence"] = _component_confidence(signals, "news_confidence", "news_sentiment")
    table["policy_sentiment"] = pd.to_numeric(signals["policy_sentiment"], errors="coerce")
    table["policy_confidence"] = _component_confidence(signals, "policy_confidence", "policy_sentiment")
    table["news_count"] = pd.to_numeric(signals["news_count"], errors="coerce").fillna(0).astype(int)
    table["news_evidence"] = signals.get("news_evidence", pd.Series("", index=signals.index)).fillna("")

    table["intraday_score"] = _robust_cross_sectional_score(table["intraday_change"])
    table["sector_score"] = _robust_cross_sectional_score(table["sector_change"])
    table["news_score"] = table["news_sentiment"].clip(-1.0, 1.0)
    table["policy_score"] = table["policy_sentiment"].clip(-1.0, 1.0)

    context_scores = []
    confidences = []
    for _, row in table.iterrows():
        weighted_sum = 0.0
        available_weight = 0.0
        for column, weight in CONTEXT_WEIGHTS.items():
            value = row[column]
            confidence_column = column.replace("_score", "_confidence")
            confidence = float(row[confidence_column])
            effective_weight = weight * confidence
            if not pd.isna(value) and effective_weight > 0:
                weighted_sum += float(value) * effective_weight
                available_weight += effective_weight
        context_scores.append(weighted_sum / available_weight if available_weight else 0.0)
        confidences.append(available_weight)

    table["context_score"] = pd.Series(context_scores, index=table.index).clip(-1.0, 1.0)
    table["context_confidence"] = pd.Series(confidences, index=table.index).clip(0.0, 1.0)
    table["context_annual_tilt"] = (
        table["context_score"] * table["context_confidence"] * max_context_annual_tilt
    )
    table["blended_expected_return"] = (
        table["historical_annual_return"] + table["context_annual_tilt"]
    )
    return ExpectedReturnForecast(
        table=table,
        context_source=context.source,
        context_fetched_at=context.fetched_at,
        context_market_data_as_of=context.market_data_as_of,
        notes=context.notes
        + (
            "Historical means are shrunk 35% toward the cross-sectional median.",
            f"Historical expected returns are clipped to {historical_return_bounds} before context tilt.",
            "Live context contributes at most +/-8 percentage points of annualized return tilt.",
            "The context-aware target is not included in historical backtest results.",
        ),
    )


def _robust_cross_sectional_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    available = numeric.dropna()
    if available.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    median = float(available.median())
    mad = float((available - median).abs().median())
    if mad <= 1e-12:
        standard = float(available.std(ddof=0))
        if standard <= 1e-12:
            scores = pd.Series(0.0, index=available.index)
        else:
            scores = (available - median) / standard
    else:
        scores = (available - median) / (1.4826 * mad)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[scores.index] = scores.clip(-2.0, 2.0) / 2.0
    return result


def _component_confidence(
    signals: pd.DataFrame,
    confidence_column: str,
    value_column: str,
) -> pd.Series:
    if confidence_column in signals:
        return pd.to_numeric(signals[confidence_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return signals[value_column].notna().astype(float)
