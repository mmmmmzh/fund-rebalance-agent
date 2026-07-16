from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.config import DEFAULT_FUND_UNIVERSE, FundSpec


@dataclass(frozen=True)
class PriceDataset:
    prices: pd.DataFrame
    fund_universe: tuple[FundSpec, ...]
    source: str


def load_price_dataset(
    source: str = "sample",
    start: str = "2020-01-01",
    end: str | None = None,
    fund_universe: Iterable[FundSpec] = DEFAULT_FUND_UNIVERSE,
) -> PriceDataset:
    universe = tuple(fund_universe)
    if source == "sample":
        prices = generate_sample_prices(universe, start=start, end=end)
    elif source == "plugin":
        loader = require_adapter("price_data", get_adapters().price_data)
        prices = loader(universe, start, end)
    else:
        raise ValueError(f"Unsupported data source: {source}")
    return PriceDataset(
        prices=_validate_prices(prices, universe),
        fund_universe=universe,
        source=source,
    )


def generate_sample_prices(
    fund_universe: Iterable[FundSpec],
    start: str = "2020-01-01",
    end: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, synthetic fund-like prices for offline validation."""

    end = end or date.today().isoformat()
    dates = pd.bdate_range(start=start, end=end)
    universe = tuple(fund_universe)
    rng = np.random.default_rng(seed)
    market_factor = rng.normal(0.00025, 0.010, len(dates))
    rate_factor = rng.normal(0.00008, 0.002, len(dates))
    commodity_factor = rng.normal(0.00012, 0.008, len(dates))
    overseas_factor = rng.normal(0.00030, 0.011, len(dates))
    returns: dict[str, np.ndarray] = {}
    for index, fund in enumerate(universe):
        noise = rng.normal(0.0, 0.004 + index * 0.0002, len(dates))
        category = fund.category.lower()
        if "bond" in category:
            daily = rate_factor + rng.normal(0.0, 0.001, len(dates))
        elif "commodity" in category:
            daily = 0.25 * market_factor + commodity_factor + noise
        elif "overseas" in category:
            daily = 0.45 * market_factor + overseas_factor + noise
        elif "defensive" in category:
            daily = 0.60 * market_factor + rng.normal(0.0, 0.005, len(dates))
        elif "growth" in category:
            daily = 1.20 * market_factor + rng.normal(0.0, 0.010, len(dates))
        else:
            daily = 0.95 * market_factor + noise
        returns[fund.code] = daily
    prices = 100.0 * (1.0 + pd.DataFrame(returns, index=dates)).cumprod()
    return prices.round(4)


def _validate_prices(prices: pd.DataFrame, universe: tuple[FundSpec, ...]) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price adapter returned no data.")
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.columns = [str(column).zfill(6) for column in frame.columns]
    expected = [fund.code for fund in universe]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Price data is missing configured fund codes: {missing}")
    frame = frame[expected].apply(pd.to_numeric, errors="coerce").sort_index().ffill().dropna()
    if len(frame) < 8 or (frame <= 0).any(axis=None):
        raise ValueError("Price data must contain at least eight positive aligned rows.")
    return frame
