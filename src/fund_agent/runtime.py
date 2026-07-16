from __future__ import annotations

from fund_agent.config import FundSpec
from fund_agent.data import PriceDataset, load_price_dataset


def load_dataset_with_fallback(
    source: str,
    start: str,
    end: str | None,
    fund_universe: tuple[FundSpec, ...] | None = None,
    no_fallback: bool = False,
) -> tuple[PriceDataset, str]:
    data_source_label = source
    try:
        return (
            load_price_dataset(
                source=source,
                start=start,
                end=end,
                fund_universe=fund_universe or (),
            )
            if fund_universe
            else load_price_dataset(source=source, start=start, end=end)
        ), data_source_label
    except Exception as exc:
        if source == "sample" or no_fallback:
            raise
        print(f"Warning: source={source!r} failed with {type(exc).__name__}: {exc}")
        print("Falling back to deterministic sample data.")
        data_source_label = f"sample fallback after {source} failure ({type(exc).__name__})"
        return (
            load_price_dataset(
                source="sample",
                start=start,
                end=end,
                fund_universe=fund_universe or (),
            )
            if fund_universe
            else load_price_dataset(source="sample", start=start, end=end)
        ), data_source_label
