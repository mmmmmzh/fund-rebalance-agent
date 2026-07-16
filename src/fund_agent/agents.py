from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fund_agent.config import FundSpec, RiskProfile
from fund_agent.metrics import compute_returns, summarize_assets
from fund_agent.return_forecast import ExpectedReturnForecast


@dataclass(frozen=True)
class AgentBrief:
    role: str
    findings: list[str]


def market_analyst_brief(prices: pd.DataFrame) -> AgentBrief:
    returns = compute_returns(prices)
    equal_weight_return = returns.mean(axis=1)
    recent = equal_weight_return.tail(63)
    trend = (1.0 + recent).prod() - 1.0
    vol = recent.std() * (252 ** 0.5)
    if trend > 0.05 and vol < 0.20:
        regime = "positive trend with moderate volatility"
    elif trend < -0.05:
        regime = "recent drawdown or weak trend"
    elif vol > 0.25:
        regime = "high-volatility market"
    else:
        regime = "range-bound or mixed market"
    return AgentBrief(
        role="Market Analyst",
        findings=[
            f"Recent 3-month equal-weight return: {trend:.2%}.",
            f"Recent annualized equal-weight volatility: {vol:.2%}.",
            f"Regime assessment: {regime}.",
        ],
    )


def fund_analyst_brief(prices: pd.DataFrame, fund_universe: tuple[FundSpec, ...]) -> AgentBrief:
    summary = summarize_assets(prices)
    best_sharpe = summary["sharpe"].idxmax()
    worst_drawdown = summary["max_drawdown"].idxmin()
    name_map = {fund.code: fund.name for fund in fund_universe}
    return AgentBrief(
        role="Fund Analyst",
        findings=[
            f"Best historical Sharpe asset: {best_sharpe} {name_map.get(best_sharpe, '')}.",
            f"Largest historical drawdown asset: {worst_drawdown} {name_map.get(worst_drawdown, '')}.",
            "ETF/index-style assets are used first to reduce style-drift ambiguity.",
        ],
    )


def risk_manager_brief(target_weights: pd.Series, fund_universe: tuple[FundSpec, ...], profile: RiskProfile) -> AgentBrief:
    equity_codes = [fund.code for fund in fund_universe if fund.is_equity_like and fund.code in target_weights.index]
    equity_weight = float(target_weights.loc[equity_codes].sum()) if equity_codes else 0.0
    max_single = float(target_weights.max())
    violations = []
    if equity_weight > profile.max_equity_weight + 1e-6:
        violations.append("equity exposure exceeds profile cap")
    if max_single > profile.max_single_asset_weight + 1e-6:
        violations.append("single asset weight exceeds profile cap")
    if not violations:
        violations.append("no hard risk-constraint violation detected")
    return AgentBrief(
        role="Risk Manager",
        findings=[
            f"Risk profile: {profile.name.value}.",
            f"Equity-like exposure: {equity_weight:.2%} / cap {profile.max_equity_weight:.2%}.",
            f"Max single asset weight: {max_single:.2%} / cap {profile.max_single_asset_weight:.2%}.",
            f"Constraint check: {', '.join(violations)}.",
        ],
    )


def return_forecast_brief(forecast: ExpectedReturnForecast) -> AgentBrief:
    table = forecast.table
    strongest = table["context_annual_tilt"].idxmax()
    weakest = table["context_annual_tilt"].idxmin()
    average_confidence = float(table["context_confidence"].mean())
    return AgentBrief(
        role="Return Forecast",
        findings=[
            f"Context source: {forecast.context_source}.",
            f"Market data as of: {forecast.context_market_data_as_of or 'unknown'}.",
            f"Average context coverage: {average_confidence:.0%}.",
            f"Strongest context tilt: {strongest} {table.loc[strongest, 'context_annual_tilt']:+.2%}.",
            f"Weakest context tilt: {weakest} {table.loc[weakest, 'context_annual_tilt']:+.2%}.",
            "Context signals are bounded inputs to a deterministic optimizer, not an automated trading decision.",
        ],
    )
