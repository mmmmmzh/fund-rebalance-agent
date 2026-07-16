from __future__ import annotations

from fund_agent.backtest import walk_forward_backtest
from fund_agent.config import DEFAULT_FUND_UNIVERSE, get_risk_profile
from fund_agent.data import generate_sample_prices
from fund_agent.optimizer import equal_weight


def test_walk_forward_backtest_generates_summary() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, start="2020-01-01", end="2023-12-31")
    profile = get_risk_profile("balanced")
    result = walk_forward_backtest(
        prices,
        DEFAULT_FUND_UNIVERSE,
        profile,
        equal_weight,
        lookback_days=126,
    )

    assert not result.portfolio_returns.empty
    assert not result.equity_curve.empty
    assert result.summary["rebalance_count"] > 0
    assert "sharpe" in result.summary


def test_daily_rebalance_has_more_rebalance_points_than_monthly() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, start="2020-01-01", end="2021-12-31")
    profile = get_risk_profile("aggressive")
    daily = walk_forward_backtest(
        prices,
        DEFAULT_FUND_UNIVERSE,
        profile,
        equal_weight,
        lookback_days=126,
        rebalance_freq="D",
    )
    monthly = walk_forward_backtest(
        prices,
        DEFAULT_FUND_UNIVERSE,
        profile,
        equal_weight,
        lookback_days=126,
        rebalance_freq="ME",
    )

    assert daily.summary["rebalance_count"] > monthly.summary["rebalance_count"]
