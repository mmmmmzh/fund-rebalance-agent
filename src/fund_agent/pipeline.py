from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fund_agent.agents import (
    AgentBrief,
    fund_analyst_brief,
    market_analyst_brief,
    return_forecast_brief,
    risk_manager_brief,
)
from fund_agent.backtest import BacktestResult, walk_forward_backtest
from fund_agent.config import RiskProfile
from fund_agent.data import PriceDataset
from fund_agent.market_context import MarketContextSnapshot
from fund_agent.optimizer import (
    equal_weight,
    fixed_allocation,
    max_sharpe,
    max_sharpe_from_expected_returns,
    min_variance,
    momentum_allocation,
    risk_parity,
)
from fund_agent.return_forecast import ExpectedReturnForecast, build_expected_return_forecast


ALLOCATORS = [
    equal_weight,
    fixed_allocation,
    momentum_allocation,
    risk_parity,
    min_variance,
    max_sharpe,
]


@dataclass(frozen=True)
class AnalysisResult:
    backtests: list[BacktestResult]
    target_weights: pd.Series
    briefs: list[AgentBrief]
    forecast: ExpectedReturnForecast | None
    target_strategy: str


def run_analysis(
    dataset: PriceDataset,
    profile: RiskProfile,
    lookback_days: int,
    rebalance_freq: str,
    market_context: MarketContextSnapshot | None = None,
) -> AnalysisResult:
    backtests = [
        walk_forward_backtest(
            dataset.prices,
            dataset.fund_universe,
            profile,
            allocator,
            lookback_days=lookback_days,
            rebalance_freq=rebalance_freq,
        )
        for allocator in ALLOCATORS
    ]

    target_window = dataset.prices.pct_change(fill_method=None).dropna().tail(lookback_days)
    forecast = None
    target_strategy = "historical_max_sharpe"
    if market_context is not None:
        forecast = build_expected_return_forecast(
            dataset.prices,
            dataset.fund_universe,
            market_context,
            lookback_days=lookback_days,
        )
        target_weights = max_sharpe_from_expected_returns(
            target_window,
            dataset.fund_universe,
            profile,
            forecast.expected_returns,
        ).weights
        target_strategy = "context_aware_max_sharpe"
    else:
        target_weights = max_sharpe(target_window, dataset.fund_universe, profile).weights
    briefs = [
        market_analyst_brief(dataset.prices),
        fund_analyst_brief(dataset.prices, dataset.fund_universe),
        risk_manager_brief(target_weights, dataset.fund_universe, profile),
    ]
    if forecast is not None:
        briefs.insert(2, return_forecast_brief(forecast))
    return AnalysisResult(
        backtests=backtests,
        target_weights=target_weights,
        briefs=briefs,
        forecast=forecast,
        target_strategy=target_strategy,
    )


def backtest_summary_frame(backtests: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in backtests:
        row = {"strategy": result.strategy, "fee_model": result.fee_model}
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows)
