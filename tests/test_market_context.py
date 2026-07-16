from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_agent.adapters import AdapterBundle, install_adapters
from fund_agent.config import DEFAULT_FUND_UNIVERSE
from fund_agent.data import generate_sample_prices
from fund_agent.market_context import MarketContextSnapshot, load_market_context
from fund_agent.return_forecast import build_expected_return_forecast


def test_sample_context_is_offline_and_deterministic_shape() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, "2024-01-01", "2025-12-31")
    context = load_market_context("sample", DEFAULT_FUND_UNIVERSE, prices)

    assert context is not None
    assert context.signals.index.tolist() == [fund.code for fund in DEFAULT_FUND_UNIVERSE]
    assert context.signals["news_count"].eq(0).all()
    assert "synthetic" in context.source


def test_plugin_context_requires_explicit_adapter() -> None:
    prices = generate_sample_prices(DEFAULT_FUND_UNIVERSE, "2024-01-01", "2025-12-31")

    with pytest.raises(RuntimeError, match="not installed"):
        load_market_context("plugin", DEFAULT_FUND_UNIVERSE, prices)


def test_expected_return_forecast_bounds_plugin_context_tilt() -> None:
    universe = DEFAULT_FUND_UNIVERSE
    prices = generate_sample_prices(universe, "2024-01-01", "2025-12-31")

    def loader(funds, _prices):
        signals = pd.DataFrame(
            {
                "intraday_change": np.linspace(-0.05, 0.05, len(funds)),
                "sector_change": np.linspace(-0.03, 0.03, len(funds)),
                "news_sentiment": np.linspace(-1.0, 1.0, len(funds)),
                "policy_sentiment": np.linspace(-1.0, 1.0, len(funds)),
                "news_count": 1,
            },
            index=[fund.code for fund in funds],
        )
        return MarketContextSnapshot(signals, "test-plugin", pd.Timestamp("2026-06-26"), "2026-06-26")

    install_adapters(AdapterBundle(market_context=loader))
    context = load_market_context("plugin", universe, prices)
    forecast = build_expected_return_forecast(prices, universe, context)

    assert forecast.table["context_annual_tilt"].abs().max() <= 0.08 + 1e-12
    assert forecast.expected_returns.notna().all()
