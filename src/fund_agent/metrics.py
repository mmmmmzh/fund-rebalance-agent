from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def annual_return(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    return returns.mean() * TRADING_DAYS


def annual_volatility(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    vol = float(annual_volatility(excess))
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(annual_return(excess) / vol)


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown_from_returns(returns))
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    return float(annual_return(returns) / mdd)


def summarize_assets(prices: pd.DataFrame) -> pd.DataFrame:
    returns = compute_returns(prices)
    rows = []
    for code in prices.columns:
        series = returns[code].dropna()
        rows.append(
            {
                "code": code,
                "annual_return": float(annual_return(series)),
                "annual_volatility": float(annual_volatility(series)),
                "max_drawdown": max_drawdown_from_returns(series),
                "sharpe": sharpe_ratio(series),
                "calmar": calmar_ratio(series),
            }
        )
    return pd.DataFrame(rows).set_index("code")


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    return returns.mul(weight_series, axis=1).sum(axis=1)


def summarize_portfolio(returns: pd.Series) -> dict[str, float]:
    return {
        "cumulative_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
        "annual_return": float(annual_return(returns)),
        "annual_volatility": float(annual_volatility(returns)),
        "max_drawdown": max_drawdown_from_returns(returns),
        "sharpe": sharpe_ratio(returns),
        "calmar": calmar_ratio(returns),
    }

