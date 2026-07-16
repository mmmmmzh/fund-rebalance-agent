from __future__ import annotations

import pandas as pd

from fund_agent.config import DEFAULT_FUND_UNIVERSE, get_risk_profile
from fund_agent.data import generate_sample_prices
from fund_agent.metrics import compute_returns, summarize_assets
from fund_agent.optimizer import fixed_allocation, max_sharpe


def test_sample_prices_and_metrics_are_non_empty() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, start="2020-01-01", end="2022-12-31")
    returns = compute_returns(prices)
    summary = summarize_assets(prices)

    assert not prices.empty
    assert not returns.empty
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) >= {"annual_return", "annual_volatility", "max_drawdown", "sharpe"}


def test_max_sharpe_respects_balanced_constraints() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, start="2020-01-01", end="2022-12-31")
    returns = compute_returns(prices)
    profile = get_risk_profile("balanced")
    allocation = max_sharpe(returns, DEFAULT_FUND_UNIVERSE, profile)

    equity_codes = [fund.code for fund in DEFAULT_FUND_UNIVERSE if fund.is_equity_like]
    equity_weight = allocation.weights.loc[equity_codes].sum()

    assert abs(allocation.weights.sum() - 1.0) < 1e-6
    assert allocation.weights.max() <= profile.max_single_asset_weight + 1e-6
    assert equity_weight <= profile.max_equity_weight + 1e-6


def test_fixed_allocation_runs_for_all_columns() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, start="2020-01-01", end="2022-12-31")
    returns = compute_returns(prices)
    profile = get_risk_profile("balanced")
    allocation = fixed_allocation(returns, DEFAULT_FUND_UNIVERSE, profile)

    assert set(allocation.weights.index) == set(returns.columns)
    assert abs(allocation.weights.sum() - 1.0) < 1e-6
