from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RiskProfileName(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class FundSpec:
    code: str
    name: str
    category: str
    is_equity_like: bool
    instrument_type: str = "etf"


@dataclass(frozen=True)
class RiskProfile:
    name: RiskProfileName
    max_equity_weight: float
    max_single_asset_weight: float
    max_turnover: float
    transaction_cost: float


DEFAULT_FUND_UNIVERSE: tuple[FundSpec, ...] = (
    FundSpec("800001", "Demo Broad Market ETF", "broad_market", True),
    FundSpec("800002", "Demo Mid Market ETF", "mid_market", True),
    FundSpec("800003", "Demo Growth ETF", "growth", True),
    FundSpec("800004", "Demo Defensive ETF", "defensive", True),
    FundSpec("800005", "Demo Bond ETF", "bond", False),
    FundSpec("800006", "Demo Commodity ETF", "commodity", False),
    FundSpec("800007", "Demo Global ETF", "overseas", True),
    FundSpec("800008", "Demo Short Bond ETF", "short_bond", False),
)


RISK_PROFILES: dict[RiskProfileName, RiskProfile] = {
    RiskProfileName.CONSERVATIVE: RiskProfile(
        name=RiskProfileName.CONSERVATIVE,
        max_equity_weight=0.35,
        max_single_asset_weight=0.25,
        max_turnover=0.20,
        transaction_cost=0.0010,
    ),
    RiskProfileName.BALANCED: RiskProfile(
        name=RiskProfileName.BALANCED,
        max_equity_weight=0.60,
        max_single_asset_weight=0.30,
        max_turnover=0.30,
        transaction_cost=0.0010,
    ),
    RiskProfileName.AGGRESSIVE: RiskProfile(
        name=RiskProfileName.AGGRESSIVE,
        max_equity_weight=0.85,
        max_single_asset_weight=0.40,
        max_turnover=0.45,
        transaction_cost=0.0010,
    ),
}


DEFAULT_RISK_PROFILE_NAME = RiskProfileName.AGGRESSIVE
DEFAULT_REBALANCE_FREQ = "D"
DEFAULT_LOOKBACK_DAYS = 252


def get_risk_profile(name: str | RiskProfileName) -> RiskProfile:
    try:
        key = RiskProfileName(str(name).lower())
    except ValueError as exc:
        valid = ", ".join(profile.value for profile in RiskProfileName)
        raise ValueError(f"Unknown risk profile {name!r}. Valid values: {valid}") from exc
    return RISK_PROFILES[key]
