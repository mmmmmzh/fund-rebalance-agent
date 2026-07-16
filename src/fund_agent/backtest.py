from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from fund_agent.config import DEFAULT_REBALANCE_FREQ, FundSpec, RiskProfile
from fund_agent.metrics import compute_returns, summarize_portfolio
from fund_agent.optimizer import AllocationResult, limit_turnover


Allocator = Callable[[pd.DataFrame, tuple[FundSpec, ...], RiskProfile], AllocationResult]


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    summary: dict[str, float]
    blocked_turnover: pd.Series
    fee_model: str


def walk_forward_backtest(
    prices: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    allocator: Allocator,
    lookback_days: int = 252,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
) -> BacktestResult:
    returns = compute_returns(prices)
    rebalance_dates = _rebalance_dates(returns, rebalance_freq)
    rebalance_dates = [dt for dt in rebalance_dates if returns.index.get_loc(dt) >= lookback_days]
    if len(rebalance_dates) < 2:
        raise ValueError(
            "Not enough data for walk-forward backtest. "
            f"price_rows={len(prices)}, return_rows={len(returns)}, "
            f"lookback_days={lookback_days}, rebalance_freq={rebalance_freq}, "
            f"usable_rebalance_points={len(rebalance_dates)}."
        )

    previous_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    realized_returns = []
    weights_records = []
    turnover_records = []
    blocked_turnover_records = []
    for idx, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[idx + 1]
        loc = returns.index.get_loc(rebalance_date)
        window = returns.iloc[loc - lookback_days:loc]
        allocation = allocator(window, fund_universe, profile)
        target_weights = limit_turnover(allocation.weights, previous_weights, profile.max_turnover)
        turnover = float((target_weights - previous_weights).abs().sum() / 2.0)

        period_returns = returns.loc[rebalance_date:next_date].iloc[1:]
        gross = period_returns.mul(target_weights, axis=1).sum(axis=1)
        if not gross.empty:
            cost = turnover * profile.transaction_cost
            gross.iloc[0] = gross.iloc[0] - cost
            realized_returns.append(gross)

        weights_records.append(target_weights.rename(rebalance_date))
        turnover_records.append((rebalance_date, turnover))
        blocked_turnover_records.append((rebalance_date, 0.0))
        previous_weights = target_weights

    portfolio = pd.concat(realized_returns).sort_index()
    weights = pd.DataFrame(weights_records)
    turnover_series = pd.Series(
        data=[value for _, value in turnover_records],
        index=[dt for dt, _ in turnover_records],
        name="turnover",
    )
    blocked_turnover_series = pd.Series(
        data=[value for _, value in blocked_turnover_records],
        index=[dt for dt, _ in blocked_turnover_records],
        name="blocked_turnover",
    )
    equity_curve = (1.0 + portfolio.fillna(0.0)).cumprod()
    summary = summarize_portfolio(portfolio)
    summary["average_turnover"] = float(turnover_series.mean())
    summary["rebalance_count"] = float(len(turnover_series))
    summary["blocked_turnover_total"] = float(blocked_turnover_series.sum())

    return BacktestResult(
        strategy=allocator.__name__,
        portfolio_returns=portfolio,
        equity_curve=equity_curve,
        weights=weights,
        turnover=turnover_series,
        summary=summary,
        blocked_turnover=blocked_turnover_series,
        fee_model="constant_turnover_rate",
    )


def _rebalance_dates(returns: pd.DataFrame, freq: str) -> list[pd.Timestamp]:
    freq = _normalize_rebalance_freq(freq)
    schedule = returns.resample(freq).last().index
    dates = []
    for timestamp in schedule:
        pos = returns.index.searchsorted(timestamp, side="right") - 1
        if pos >= 0:
            dates.append(returns.index[pos])
    return sorted(set(dates))


def _normalize_rebalance_freq(freq: str) -> str:
    normalized = freq.strip().lower()
    aliases = {
        "daily": "D",
        "day": "D",
        "d": "D",
        "trading_day": "D",
        "trading-day": "D",
        "monthly": "ME",
        "month": "ME",
        "m": "ME",
        "me": "ME",
        "quarterly": "QE",
        "quarter": "QE",
        "q": "QE",
        "qe": "QE",
    }
    return aliases.get(normalized, freq)
