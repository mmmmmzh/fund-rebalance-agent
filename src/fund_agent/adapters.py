from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from fund_agent.config import FundSpec


class PriceDataAdapter(Protocol):
    def __call__(
        self,
        universe: tuple[FundSpec, ...],
        start: str,
        end: str | None,
    ) -> pd.DataFrame: ...


class MarketContextAdapter(Protocol):
    def __call__(
        self,
        universe: tuple[FundSpec, ...],
        prices: pd.DataFrame,
    ) -> Any: ...


class CandidateSourceAdapter(Protocol):
    def __call__(self, limit: int) -> pd.DataFrame: ...


class ResearchSourceAdapter(Protocol):
    def __call__(
        self,
        topics: Iterable[Any],
        universe: tuple[FundSpec, ...],
        as_of: date,
    ) -> list[dict[str, Any]]: ...


class SkillAdapter(Protocol):
    def __call__(
        self,
        skill_id: str,
        facts: pd.DataFrame,
        settings: dict[str, Any],
    ) -> dict[str, dict[str, Any]]: ...


class TradingCalendarAdapter(Protocol):
    def __call__(self, start: date, end: date) -> set[date]: ...


@dataclass(frozen=True)
class AdapterBundle:
    """Optional private integrations installed explicitly by a host application.

    The public package ships no network implementation. Callers can install a
    bundle at process startup; otherwise every workflow remains deterministic
    and offline.
    """

    price_data: PriceDataAdapter | None = None
    market_context: MarketContextAdapter | None = None
    candidates: CandidateSourceAdapter | None = None
    research: ResearchSourceAdapter | None = None
    skill: SkillAdapter | None = None
    trading_calendar: TradingCalendarAdapter | None = None


_ADAPTERS = AdapterBundle()


def install_adapters(bundle: AdapterBundle) -> None:
    global _ADAPTERS
    _ADAPTERS = bundle


def get_adapters() -> AdapterBundle:
    return _ADAPTERS


def reset_adapters() -> None:
    install_adapters(AdapterBundle())


def require_adapter(name: str, adapter: Callable[..., Any] | None) -> Callable[..., Any]:
    if adapter is None:
        raise RuntimeError(
            f"Adapter capability {name!r} is not installed. "
            "The public edition intentionally ships offline implementations only."
        )
    return adapter
