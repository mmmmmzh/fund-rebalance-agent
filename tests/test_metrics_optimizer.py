from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from fund_agent.backtest import walk_forward_backtest
from fund_agent.config import DEFAULT_FUND_UNIVERSE, FundSpec, RiskProfile, get_risk_profile
from fund_agent.data import generate_sample_prices
from fund_agent.metrics import compute_returns, summarize_assets
from fund_agent.optimizer import (
    AllocationResult,
    apply_constraints,
    equal_weight,
    fixed_allocation,
    max_sharpe,
    min_variance,
    momentum_allocation,
    risk_parity,
)


Allocator = Callable[[pd.DataFrame, tuple[FundSpec, ...], RiskProfile], AllocationResult]
ALLOCATORS: tuple[Allocator, ...] = (
    equal_weight,
    fixed_allocation,
    momentum_allocation,
    min_variance,
    max_sharpe,
    risk_parity,
)
PROFILE_NAMES = ("conservative", "balanced", "aggressive")


@pytest.fixture(scope="module")
def sample_prices() -> pd.DataFrame:
    return generate_sample_prices(
        DEFAULT_FUND_UNIVERSE,
        start="2021-01-01",
        end="2023-12-31",
    )


@pytest.fixture(scope="module")
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    return compute_returns(sample_prices)


def _assert_constraints(
    weights: pd.Series,
    profile: RiskProfile,
    universe: tuple[FundSpec, ...] = DEFAULT_FUND_UNIVERSE,
) -> None:
    equity_codes = [fund.code for fund in universe if fund.is_equity_like]
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert weights.min() >= -1e-8
    assert weights.max() <= profile.max_single_asset_weight + 1e-7
    assert weights.reindex(equity_codes).fillna(0.0).sum() <= profile.max_equity_weight + 1e-7


def test_sample_prices_and_metrics_are_non_empty(sample_prices: pd.DataFrame) -> None:
    returns = compute_returns(sample_prices)
    summary = summarize_assets(sample_prices)

    assert not sample_prices.empty
    assert not returns.empty
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) >= {
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "sharpe",
    }


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
@pytest.mark.parametrize("allocator", ALLOCATORS, ids=lambda allocator: allocator.__name__)
def test_all_allocators_respect_all_risk_profiles(
    sample_returns: pd.DataFrame,
    profile_name: str,
    allocator: Allocator,
) -> None:
    profile = get_risk_profile(profile_name)
    allocation = allocator(sample_returns, DEFAULT_FUND_UNIVERSE, profile)

    assert allocation.weights.index.tolist() == sample_returns.columns.tolist()
    _assert_constraints(allocation.weights, profile)


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_concentrated_weights_are_projected_to_feasible_portfolio(profile_name: str) -> None:
    profile = get_risk_profile(profile_name)
    weights = pd.Series(0.0, index=[fund.code for fund in DEFAULT_FUND_UNIVERSE])
    weights.iloc[0] = 1.0

    projected = apply_constraints(weights, DEFAULT_FUND_UNIVERSE, profile)

    _assert_constraints(projected, profile)


def test_infeasible_conservative_universe_is_rejected() -> None:
    universe_with_two_non_equity_assets = DEFAULT_FUND_UNIVERSE[:-1]
    profile = get_risk_profile("conservative")
    weights = pd.Series(
        1.0 / len(universe_with_two_non_equity_assets),
        index=[fund.code for fund in universe_with_two_non_equity_assets],
    )

    with pytest.raises(
        ValueError,
        match=r"约束不可行: assets=7, non_equity_assets=2",
    ):
        apply_constraints(weights, universe_with_two_non_equity_assets, profile)


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
@pytest.mark.parametrize("allocator", ALLOCATORS, ids=lambda allocator: allocator.__name__)
def test_every_backtest_weight_row_respects_constraints(
    sample_prices: pd.DataFrame,
    profile_name: str,
    allocator: Allocator,
) -> None:
    profile = get_risk_profile(profile_name)
    result = walk_forward_backtest(
        sample_prices,
        DEFAULT_FUND_UNIVERSE,
        profile,
        allocator,
        lookback_days=63,
        rebalance_freq="QE",
    )

    assert not result.weights.empty
    for _, weights in result.weights.iterrows():
        _assert_constraints(weights, profile)
