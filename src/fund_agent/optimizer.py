from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fund_agent.config import FundSpec, RiskProfile


@dataclass(frozen=True)
class AllocationResult:
    strategy: str
    weights: pd.Series
    notes: list[str]


def equal_weight(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> AllocationResult:
    weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    weights = apply_constraints(weights, fund_universe, profile)
    return AllocationResult("equal_weight", weights, ["Equal initial weights, then risk constraints."])


def fixed_allocation(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    equity_target: float | None = None,
) -> AllocationResult:
    equity_target = min(equity_target if equity_target is not None else profile.max_equity_weight, profile.max_equity_weight)
    equity_codes = [fund.code for fund in fund_universe if fund.is_equity_like and fund.code in returns.columns]
    defensive_codes = [code for code in returns.columns if code not in equity_codes]

    weights = pd.Series(0.0, index=returns.columns)
    if equity_codes:
        weights.loc[equity_codes] = equity_target / len(equity_codes)
    if defensive_codes:
        weights.loc[defensive_codes] = (1.0 - equity_target) / len(defensive_codes)
    weights = apply_constraints(weights, fund_universe, profile)
    return AllocationResult("fixed_allocation", weights, [f"Target equity-like exposure: {equity_target:.0%}."])


def momentum_allocation(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    top_n: int = 3,
) -> AllocationResult:
    lookback_return = (1.0 + returns.fillna(0.0)).prod() - 1.0
    selected = lookback_return.sort_values(ascending=False).head(top_n).index
    weights = pd.Series(0.0, index=returns.columns)
    weights.loc[selected] = 1.0 / len(selected)
    weights = apply_constraints(weights, fund_universe, profile)
    return AllocationResult(
        "momentum",
        weights,
        [f"Selected top {len(selected)} assets by lookback total return."],
    )


def min_variance(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> AllocationResult:
    cov = returns.cov().values * 252
    n_assets = len(returns.columns)
    bounds = [(0.0, profile.max_single_asset_weight) for _ in range(n_assets)]
    constraints = _constraints(returns.columns, fund_universe, profile)
    init = np.repeat(1.0 / n_assets, n_assets)

    result = minimize(
        lambda w: float(w @ cov @ w),
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = pd.Series(result.x if result.success else init, index=returns.columns)
    weights = apply_constraints(weights, fund_universe, profile)
    notes = ["Minimized annualized variance under long-only risk constraints."]
    if not result.success:
        notes.append(f"Optimizer fallback used: {result.message}")
    return AllocationResult("min_variance", weights, notes)


def max_sharpe(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> AllocationResult:
    mean = returns.mean() * 252
    return max_sharpe_from_expected_returns(returns, fund_universe, profile, mean)


def max_sharpe_from_expected_returns(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    expected_returns: pd.Series,
) -> AllocationResult:
    mean = expected_returns.reindex(returns.columns).fillna(0.0).values
    cov = returns.cov().values * 252
    n_assets = len(returns.columns)
    bounds = [(0.0, profile.max_single_asset_weight) for _ in range(n_assets)]
    constraints = _constraints(returns.columns, fund_universe, profile)
    init = np.repeat(1.0 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        ret = float(weights @ mean)
        vol = float(np.sqrt(weights @ cov @ weights))
        if vol <= 0 or np.isnan(vol):
            return 1e6
        return -ret / vol

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = pd.Series(result.x if result.success else init, index=returns.columns)
    weights = apply_constraints(weights, fund_universe, profile)
    notes = ["Maximized supplied expected-return Sharpe ratio under long-only risk constraints."]
    if not result.success:
        notes.append(f"Optimizer fallback used: {result.message}")
    return AllocationResult("max_sharpe", weights, notes)


def risk_parity(
    returns: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> AllocationResult:
    vol = returns.std(ddof=1).replace(0, np.nan)
    inverse_vol = 1.0 / vol
    weights = inverse_vol / inverse_vol.sum()
    weights = weights.fillna(1.0 / len(returns.columns))
    weights = apply_constraints(weights, fund_universe, profile)
    return AllocationResult("risk_parity", weights, ["Inverse-volatility approximation to risk parity."])


def apply_constraints(
    weights: pd.Series,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> pd.Series:
    weights = weights.clip(lower=0.0, upper=profile.max_single_asset_weight)
    weights = _cap_equity_exposure(weights, fund_universe, profile.max_equity_weight)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def limit_turnover(target: pd.Series, previous: pd.Series, max_turnover: float) -> pd.Series:
    previous = previous.reindex(target.index).fillna(0.0)
    target = target.reindex(previous.index).fillna(0.0)
    turnover = float((target - previous).abs().sum() / 2.0)
    if turnover <= max_turnover or turnover == 0:
        return target
    blend = max_turnover / turnover
    adjusted = previous + blend * (target - previous)
    return adjusted / adjusted.sum()


def _constraints(
    columns: pd.Index,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> list[dict[str, object]]:
    equity_codes = {fund.code for fund in fund_universe if fund.is_equity_like}
    equity_mask = np.array([1.0 if code in equity_codes else 0.0 for code in columns])
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: profile.max_equity_weight - float(w @ equity_mask)},
    ]


def _cap_equity_exposure(
    weights: pd.Series,
    fund_universe: tuple[FundSpec, ...],
    max_equity_weight: float,
) -> pd.Series:
    equity_codes = [fund.code for fund in fund_universe if fund.is_equity_like and fund.code in weights.index]
    defensive_codes = [code for code in weights.index if code not in equity_codes]
    equity_weight = float(weights.loc[equity_codes].sum()) if equity_codes else 0.0
    if equity_weight <= max_equity_weight or not defensive_codes:
        return weights

    scaled = weights.copy()
    scaled.loc[equity_codes] *= max_equity_weight / equity_weight
    defensive_total = float(scaled.loc[defensive_codes].sum())
    target_defensive = 1.0 - max_equity_weight
    if defensive_total > 0:
        scaled.loc[defensive_codes] *= target_defensive / defensive_total
    else:
        scaled.loc[defensive_codes] = target_defensive / len(defensive_codes)
    return scaled
