from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

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
    init = _feasible_initial_weights(returns.columns, fund_universe, profile).to_numpy()

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
    init = _feasible_initial_weights(returns.columns, fund_universe, profile).to_numpy()

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
    if weights.empty:
        raise ValueError("Cannot apply constraints to an empty portfolio.")
    if not weights.index.is_unique:
        raise ValueError("Portfolio asset codes must be unique.")
    raw_weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_weights).all():
        raise ValueError("Portfolio weights must be finite numbers.")

    _validate_constraint_feasibility(weights.index, fund_universe, profile)
    feasible_initial = _feasible_initial_weights(weights.index, fund_universe, profile)
    bounds = [(0.0, profile.max_single_asset_weight)] * len(weights)
    constraints = _constraints(weights.index, fund_universe, profile)
    result = minimize(
        lambda candidate: float(np.square(candidate - raw_weights).sum()),
        feasible_initial.to_numpy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Constraint projection failed: {result.message}")

    projected = pd.Series(result.x, index=weights.index, dtype=float)
    _assert_constraints(projected, fund_universe, profile)
    return projected


def limit_turnover(
    target: pd.Series,
    previous: pd.Series,
    max_turnover: float,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> pd.Series:
    if max_turnover < 0.0:
        raise ValueError("max_turnover must be non-negative.")
    target = target.astype(float)
    previous = previous.reindex(target.index)
    _assert_constraints(target, fund_universe, profile)
    _assert_constraints(previous, fund_universe, profile)
    turnover = float((target - previous).abs().sum() / 2.0)
    if turnover <= max_turnover or turnover == 0:
        return target.copy()
    blend = max_turnover / turnover
    adjusted = previous + blend * (target - previous)
    _assert_constraints(adjusted, fund_universe, profile)
    return adjusted


def _constraints(
    columns: pd.Index,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> list[dict[str, object]]:
    equity_mask = _equity_mask(columns, fund_universe)
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: profile.max_equity_weight - float(w @ equity_mask)},
    ]


def _validate_constraint_feasibility(
    columns: pd.Index,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> None:
    asset_count = len(columns)
    equity_mask = _equity_mask(columns, fund_universe)
    non_equity_count = int(asset_count - equity_mask.sum())
    max_single = profile.max_single_asset_weight
    required_non_equity = 1.0 - profile.max_equity_weight
    tolerance = 1e-10
    invalid_limits = not (0.0 < max_single <= 1.0 and 0.0 <= profile.max_equity_weight <= 1.0)
    insufficient_total = asset_count * max_single < 1.0 - tolerance
    insufficient_non_equity = (
        non_equity_count * max_single < required_non_equity - tolerance
    )
    if asset_count == 0 or invalid_limits or insufficient_total or insufficient_non_equity:
        raise ValueError(
            "约束不可行: "
            f"assets={asset_count}, non_equity_assets={non_equity_count}, "
            f"max_single_asset_weight={max_single:.4f}, "
            f"max_equity_weight={profile.max_equity_weight:.4f}, "
            f"total_capacity={asset_count * max_single:.4f}, "
            f"non_equity_capacity={non_equity_count * max_single:.4f}, "
            f"required_non_equity={required_non_equity:.4f}."
        )


def _feasible_initial_weights(
    columns: pd.Index,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
) -> pd.Series:
    _validate_constraint_feasibility(columns, fund_universe, profile)
    asset_count = len(columns)
    equity_mask = _equity_mask(columns, fund_universe)
    result = linprog(
        c=np.zeros(asset_count),
        A_ub=np.atleast_2d(equity_mask),
        b_ub=np.array([profile.max_equity_weight]),
        A_eq=np.ones((1, asset_count)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, profile.max_single_asset_weight)] * asset_count,
        method="highs",
    )
    if not result.success:
        raise ValueError(
            "约束不可行: linear feasibility search failed; "
            f"assets={asset_count}, max_single_asset_weight="
            f"{profile.max_single_asset_weight:.4f}, "
            f"max_equity_weight={profile.max_equity_weight:.4f}, "
            f"reason={result.message}."
        )
    feasible = pd.Series(result.x, index=columns, dtype=float)
    _assert_constraints(feasible, fund_universe, profile)
    return feasible


def _assert_constraints(
    weights: pd.Series,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    tolerance: float = 1e-7,
) -> None:
    if weights.empty or not weights.index.is_unique:
        raise ValueError("Portfolio weights must be non-empty and uniquely indexed.")
    values = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Portfolio weights must be finite numbers.")
    total = float(values.sum())
    minimum = float(values.min())
    maximum = float(values.max())
    equity_weight = float(values @ _equity_mask(weights.index, fund_universe))
    violations = []
    if abs(total - 1.0) > tolerance:
        violations.append(f"sum={total:.12f}")
    if minimum < -tolerance:
        violations.append(f"min={minimum:.12f}")
    if maximum > profile.max_single_asset_weight + tolerance:
        violations.append(
            f"max={maximum:.12f} > single_limit={profile.max_single_asset_weight:.12f}"
        )
    if equity_weight > profile.max_equity_weight + tolerance:
        violations.append(
            f"equity={equity_weight:.12f} > equity_limit={profile.max_equity_weight:.12f}"
        )
    if violations:
        raise ValueError("Portfolio violates risk constraints: " + "; ".join(violations))


def _equity_mask(
    columns: pd.Index,
    fund_universe: tuple[FundSpec, ...],
) -> np.ndarray:
    equity_codes = {fund.code for fund in fund_universe if fund.is_equity_like}
    return np.array([1.0 if str(code) in equity_codes else 0.0 for code in columns])
