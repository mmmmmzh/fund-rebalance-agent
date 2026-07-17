from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_agent.config import FundSpec
from fund_agent.data import _validate_prices


UNIVERSE = (
    FundSpec("000001", "Demo One", "bond", False),
    FundSpec("000002", "Demo Two", "broad_market", True),
)


def _prices(rows: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    return pd.DataFrame(
        {
            "000001": np.linspace(1.0, 1.2, rows),
            "000002": np.linspace(2.0, 2.4, rows),
        },
        index=dates,
    )


def test_duplicate_dates_are_rejected_after_date_normalization() -> None:
    frame = _prices()
    dates = frame.index.to_list()
    dates[4] = dates[3] + pd.Timedelta(hours=12)
    frame.index = dates

    with pytest.raises(ValueError, match="duplicate dates"):
        _validate_prices(frame, UNIVERSE)


def test_invalid_dates_are_rejected_as_nat() -> None:
    frame = _prices()
    frame.index = ["not-a-date", *frame.index[1:].astype(str)]

    with pytest.raises(ValueError, match="NaT"):
        _validate_prices(frame, UNIVERSE)


def test_timezone_aware_dates_are_rejected() -> None:
    frame = _prices()
    frame.index = frame.index.tz_localize("Asia/Shanghai")

    with pytest.raises(ValueError, match="timezone-naive"):
        _validate_prices(frame, UNIVERSE)


def test_fund_code_collision_after_zero_padding_is_rejected() -> None:
    dates = pd.bdate_range("2025-01-02", periods=30)
    frame = pd.DataFrame(
        np.column_stack(
            [
                np.linspace(1.0, 1.2, 30),
                np.linspace(1.1, 1.3, 30),
                np.linspace(2.0, 2.4, 30),
            ]
        ),
        index=dates,
        columns=[1, "000001", "000002"],
    )

    with pytest.raises(ValueError, match="duplicate fund codes after normalization"):
        _validate_prices(frame, UNIVERSE)


def test_infinite_value_is_treated_as_a_short_missing_gap() -> None:
    frame = _prices()
    previous = frame.iloc[4, 0]
    frame.iloc[5, 0] = np.inf

    result = _validate_prices(frame, UNIVERSE)

    assert result.iloc[5, 0] == previous
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("invalid_price", [0.0, -0.01])
def test_nonpositive_prices_are_rejected(invalid_price: float) -> None:
    frame = _prices()
    frame.iloc[5, 0] = invalid_price

    with pytest.raises(ValueError, match="strictly positive"):
        _validate_prices(frame, UNIVERSE)


def test_missing_configured_fund_code_is_rejected() -> None:
    frame = _prices().drop(columns="000002")

    with pytest.raises(ValueError, match="missing configured fund codes.*000002"):
        _validate_prices(frame, UNIVERSE)


def test_unsorted_dates_are_sorted_and_columns_follow_universe() -> None:
    frame = _prices().iloc[::-1].loc[:, ["000002", "000001"]]

    result = _validate_prices(frame, UNIVERSE)

    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.columns.tolist() == ["000001", "000002"]


def test_short_internal_gap_is_forward_filled() -> None:
    frame = _prices()
    previous = frame.iloc[4, 0]
    frame.iloc[5:8, 0] = np.nan

    result = _validate_prices(frame, UNIVERSE)

    assert result.iloc[5:8, 0].tolist() == [previous, previous, previous]


def test_gap_longer_than_three_rows_is_rejected() -> None:
    frame = _prices()
    frame.iloc[5:9, 0] = np.nan

    with pytest.raises(ValueError, match="gaps longer than 3"):
        _validate_prices(frame, UNIVERSE)


def test_leading_gap_is_not_backfilled_from_future_data() -> None:
    frame = _prices()
    frame.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="leading gaps"):
        _validate_prices(frame, UNIVERSE)


def test_excessive_missing_rate_is_rejected() -> None:
    frame = _prices()
    frame.iloc[[2, 5, 8, 11, 14, 17, 20], 0] = np.nan

    with pytest.raises(ValueError, match="missing rate exceeds 20%"):
        _validate_prices(frame, UNIVERSE)


def test_latest_date_must_have_real_values_for_every_fund() -> None:
    frame = _prices()
    frame.iloc[-1, 1] = np.nan

    with pytest.raises(ValueError, match="real value on the latest date"):
        _validate_prices(frame, UNIVERSE)
