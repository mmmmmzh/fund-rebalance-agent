from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.config import DEFAULT_FUND_UNIVERSE, FundSpec


MIN_PRICE_ROWS = 8
MAX_FORWARD_FILL_GAP = 3
MAX_MISSING_RATE = 0.20


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

    frame.index = _normalized_price_dates(frame.index)
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique().strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Price data contains duplicate dates: {duplicates}")

    normalized_columns = [_normalize_fund_code(column) for column in frame.columns]
    duplicate_codes = pd.Index(normalized_columns)[pd.Index(normalized_columns).duplicated()]
    if not duplicate_codes.empty:
        raise ValueError(
            "Price data contains duplicate fund codes after normalization: "
            f"{sorted(set(duplicate_codes))}"
        )
    frame.columns = normalized_columns
    expected = [_normalize_fund_code(fund.code) for fund in universe]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Price data is missing configured fund codes: {missing}")

    frame = frame.loc[:, expected].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
    missing_rates = frame.isna().mean()
    excessive_missing = missing_rates[missing_rates > MAX_MISSING_RATE]
    if not excessive_missing.empty:
        details = {code: round(float(rate), 4) for code, rate in excessive_missing.items()}
        raise ValueError(
            f"Price data missing rate exceeds {MAX_MISSING_RATE:.0%}: {details}"
        )

    latest_date = frame.index.max()
    stale_codes = [
        code
        for code in expected
        if frame[code].last_valid_index() is None
        or frame[code].last_valid_index() != latest_date
    ]
    if stale_codes:
        raise ValueError(
            "Price data must contain a real value on the latest date for every fund: "
            f"{stale_codes}"
        )

    frame = frame.ffill(limit=MAX_FORWARD_FILL_GAP)
    if frame.isna().any(axis=None):
        affected = frame.columns[frame.isna().any()].tolist()
        raise ValueError(
            "Price data contains leading gaps or gaps longer than "
            f"{MAX_FORWARD_FILL_GAP} rows: {affected}"
        )
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Price data must contain only finite numeric values.")
    if (values <= 0).any():
        raise ValueError("Price data prices must be strictly positive.")
    if len(frame) < MIN_PRICE_ROWS:
        raise ValueError(f"Price data must contain at least {MIN_PRICE_ROWS} aligned rows.")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Price data index must be unique and monotonically increasing.")
    return frame.loc[:, expected]


def _normalized_price_dates(index: pd.Index) -> pd.DatetimeIndex:
    try:
        parsed = pd.to_datetime(index, errors="coerce", format="mixed")
        dates = parsed if isinstance(parsed, pd.DatetimeIndex) else pd.DatetimeIndex(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Price data index must contain timezone-naive dates.") from exc
    if dates.isna().any():
        raise ValueError("Price data index contains invalid or missing dates (NaT).")
    if dates.tz is not None:
        raise ValueError("Price data index must use timezone-naive trading dates.")
    return dates.normalize()


def _normalize_fund_code(value: object) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        value = int(value)
    return str(value).strip().zfill(6)
