from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from fund_agent.config import DEFAULT_FUND_UNIVERSE


PROJECT_ROOT_ENV = "FUND_AGENT_PROJECT_ROOT"


def _default_project_root() -> Path:
    configured = os.getenv(PROJECT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").exists() or (current / "config").exists():
        return current
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").exists():
        return source_root
    return current


PROJECT_ROOT = _default_project_root()
CONFIG_ROOT = PROJECT_ROOT / "config"
DEMO_DATA_ROOT = PROJECT_ROOT / "data" / "demo"
DEFAULT_USER_DATA_ROOT = PROJECT_ROOT / "user_data"
USER_DATA_ROOT_ENV = "FUND_AGENT_USER_DATA_ROOT"

WRITABLE_PROFILE_FIELDS = {
    "candidate_watchlist_file",
    "investable_universe_file",
    "paper_ledger_file",
    "signal_history_file",
    "validation_dir",
    "output_root",
    "state_dir",
    "skills_dir",
}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def user_data_roots() -> tuple[Path, ...]:
    roots = [DEFAULT_USER_DATA_ROOT.resolve()]
    configured = os.getenv(USER_DATA_ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def is_user_data_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    candidate = Path(path).expanduser().resolve()
    return any(_is_within(candidate, root) for root in user_data_roots())


def _resolve_workspace(workspace: str | Path) -> Path:
    candidate = Path(workspace).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not is_user_data_path(resolved):
        roots = ", ".join(str(root) for root in user_data_roots())
        raise ValueError(f"User workspace must stay inside an allowed user_data root: {roots}")
    return resolved


class ScheduleSettings(BaseModel):
    daily_time: str = "14:30"
    weekly_weekday: Literal["friday"] = "friday"
    weekly_time: str = "14:30"
    monthly_time: str = "20:00"

    @field_validator("daily_time", "weekly_time", "monthly_time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("Time must use HH:MM 24-hour format.")
        return value


class ScreeningSettings(BaseModel):
    portfolio_gap_mode: Literal["diversification_only"] = "diversification_only"
    candidate_top_n: int = Field(default=5, ge=1, le=50)
    max_new_funds: int = Field(default=2, ge=1, le=10)
    max_new_position_weight: float = Field(default=0.10, gt=0.0, le=0.50)
    max_per_category: int = Field(default=3, ge=1, le=10)
    max_holding_days: int = Field(default=30, ge=0, le=3650)
    deep_risk: bool = True
    deep_start: str = "2023-01-01"


class MarketActionSettings(BaseModel):
    decision_threshold: float = Field(default=0.25, ge=0.05, le=0.80)
    minimum_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    risk_exit_threshold: float = Field(default=0.65, ge=0.20, le=1.0)

    @model_validator(mode="after")
    def validate_signal_range(self) -> "MarketActionSettings":
        if self.risk_exit_threshold < self.decision_threshold:
            raise ValueError("risk_exit_threshold cannot be below decision_threshold")
        return self


class InvestmentPolicySettings(BaseModel):
    horizon_min_months: int = Field(default=12, ge=1, le=120)
    horizon_max_months: int = Field(default=60, ge=1, le=240)
    max_acceptable_drawdown: float = Field(default=0.20, gt=0.0, le=0.80)
    allow_cash_reserve: bool = True
    minimum_cash_weight: float = Field(default=0.0, ge=0.0, le=0.95)
    maximum_cash_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    benchmark_mode: Literal["current_mix", "equal_weight"] = "current_mix"

    @model_validator(mode="after")
    def validate_policy_range(self) -> "InvestmentPolicySettings":
        if self.horizon_min_months > self.horizon_max_months:
            raise ValueError("horizon_min_months cannot exceed horizon_max_months")
        if self.minimum_cash_weight > self.maximum_cash_weight:
            raise ValueError("minimum_cash_weight cannot exceed maximum_cash_weight")
        if not self.allow_cash_reserve and self.minimum_cash_weight > 0:
            raise ValueError("minimum_cash_weight requires allow_cash_reserve")
        return self


class ValidationSettings(BaseModel):
    paper_trial_min_weeks: int = Field(default=8, ge=1, le=52)
    paper_trial_max_weeks: int = Field(default=12, ge=1, le=104)
    evaluation_horizons_days: list[int] = Field(default_factory=lambda: [5, 20, 60])
    api_budget_min_yuan: float = Field(default=0.0, ge=0.0, le=10000.0)
    api_budget_max_yuan: float = Field(default=50.0, ge=0.0, le=10000.0)

    @model_validator(mode="after")
    def validate_validation_range(self) -> "ValidationSettings":
        if self.paper_trial_min_weeks > self.paper_trial_max_weeks:
            raise ValueError("paper_trial_min_weeks cannot exceed paper_trial_max_weeks")
        if self.api_budget_min_yuan > self.api_budget_max_yuan:
            raise ValueError("api_budget_min_yuan cannot exceed api_budget_max_yuan")
        normalized = sorted(set(self.evaluation_horizons_days))
        if not normalized or normalized[0] < 1 or normalized[-1] > 504:
            raise ValueError("evaluation_horizons_days must be between 1 and 504")
        self.evaluation_horizons_days = normalized
        return self


class ResearchTopic(BaseModel):
    name: str
    board_type: Literal["industry", "concept"]
    board_name: str
    fund_categories: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    schema_version: int = 1
    profile_name: str = "default"
    timezone: str = "Asia/Shanghai"
    risk_profile: Literal["conservative", "balanced", "aggressive"] = "aggressive"
    data_source: Literal["sample", "plugin"] = "sample"
    market_context_source: Literal["sample", "none", "plugin"] = "sample"
    calendar_source: Literal["weekdays", "plugin"] = "weekdays"
    start_date: str = "2023-01-01"
    lookback_days: int = Field(default=60, ge=20, le=504)
    rebalance_freq: Literal["D", "ME", "QE"] = "D"
    universe_file: str = "universe.csv"
    portfolio_file: str = "portfolio.csv"
    candidate_watchlist_file: str = "candidate_watchlist.csv"
    investable_universe_file: str = "investable_universe.csv"
    paper_ledger_file: str = "paper_trades.csv"
    signal_history_file: str = "signal_history.csv"
    validation_dir: str = "validation"
    output_root: str = "runs"
    state_dir: str = ".agent"
    skills_dir: str = "skills"
    enabled_skills: list[str] = Field(
        default_factory=lambda: [
            "momentum_confirmation",
            "drawdown_guard",
            "market_context_consensus",
        ]
    )
    human_approval_required: bool = True
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)
    market_actions: MarketActionSettings = Field(default_factory=MarketActionSettings)
    investment_policy: InvestmentPolicySettings = Field(default_factory=InvestmentPolicySettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    research_topics: list[ResearchTopic] = Field(default_factory=list)


@dataclass(frozen=True)
class LoadedProfile:
    profile: UserProfile
    path: Path

    @property
    def root(self) -> Path:
        return self.path.parent

    def resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        resolved = path.expanduser().resolve() if path.is_absolute() else (self.root / path).resolve()
        allowed = _is_within(resolved, PROJECT_ROOT) or is_user_data_path(resolved)
        if not allowed:
            raise ValueError(f"Profile path escapes the project and user_data roots: {value!r}")
        return resolved


DEFAULT_RESEARCH_TOPICS: list[ResearchTopic] = [
    ResearchTopic(
        name="宽基风格",
        board_type="concept",
        board_name="broad-market-synthetic",
        fund_categories=["broad_market"],
    ),
    ResearchTopic(
        name="成长风格",
        board_type="concept",
        board_name="growth-synthetic",
        fund_categories=["growth"],
    ),
    ResearchTopic(
        name="防御风格",
        board_type="concept",
        board_name="defensive-synthetic",
        fund_categories=["defensive"],
    ),
]


RUNTIME_CSV_SCHEMAS = {
    "portfolio_file": [
        "code",
        "name",
        "current_weight",
        "current_value_yuan",
        "notes",
    ],
    "candidate_watchlist_file": [
        "code",
        "name",
        "category",
        "score",
        "first_seen",
        "last_seen",
        "status",
    ],
    "investable_universe_file": [
        "code",
        "name",
        "category",
        "is_equity_like",
        "instrument_type",
        "status",
        "notes",
    ],
    "paper_ledger_file": [
        "signal_id",
        "run_id",
        "generated_at",
        "code",
        "name",
        "signal",
        "signal_score",
        "signal_confidence",
        "reference_nav_date",
        "reference_nav",
        "review_status",
        "fund_return_5d",
        "signal_return_5d",
        "fund_return_20d",
        "signal_return_20d",
        "fund_return_60d",
        "signal_return_60d",
    ],
    "signal_history_file": [
        "run_id",
        "generated_at",
        "code",
        "name",
        "signal",
        "signal_score",
        "signal_confidence",
        "review_status",
    ],
}


def load_user_profile(path: str | Path) -> LoadedProfile:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    profile_path = candidate.resolve()
    if not (_is_within(profile_path, CONFIG_ROOT) or is_user_data_path(profile_path)):
        raise ValueError("Profiles may only be loaded from config/ or an allowed user_data root.")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    loaded = LoadedProfile(UserProfile.model_validate(payload), profile_path)
    for field_name in WRITABLE_PROFILE_FIELDS:
        try:
            resolved = loaded.resolve(getattr(loaded.profile, field_name))
        except ValueError as exc:
            raise ValueError(f"{field_name}: {exc}") from exc
        if resolved is not None and not is_user_data_path(resolved):
            raise ValueError(f"{field_name} must stay inside an allowed user_data root.")
    return loaded


def ensure_user_runtime_files(loaded: LoadedProfile) -> list[Path]:
    created: list[Path] = []
    for field_name, columns in RUNTIME_CSV_SCHEMAS.items():
        path = loaded.resolve(getattr(loaded.profile, field_name))
        if path is None or path.exists():
            continue
        if not is_user_data_path(path):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
        created.append(path)
    return created


def initialize_user_workspace(
    workspace: str | Path,
    profile_name: str,
    force: bool = False,
) -> Path:
    root = _resolve_workspace(workspace)
    profile_path = root / "profile.json"
    if profile_path.exists() and not force:
        raise FileExistsError(f"Profile already exists: {profile_path}. Use --force to replace it.")
    root.mkdir(parents=True, exist_ok=True)

    universe_rows = [
        {
            "code": fund.code,
            "name": fund.name,
            "category": fund.category,
            "is_equity_like": str(fund.is_equity_like).lower(),
            "instrument_type": fund.instrument_type,
            "status": "example",
            "notes": "Synthetic demo entry; not a real financial instrument.",
        }
        for fund in DEFAULT_FUND_UNIVERSE
    ]
    pd.DataFrame(universe_rows).to_csv(root / "universe.csv", index=False, encoding="utf-8")
    demo_weights = [0.20, 0.15, 0.10, 0.15, 0.20, 0.10, 0.10]
    portfolio_rows = [
        {
            "code": fund.code,
            "name": fund.name,
            "current_weight": weight,
            "current_value_yuan": None,
            "notes": "Not a real holding.",
        }
        for fund, weight in zip(DEFAULT_FUND_UNIVERSE, demo_weights, strict=True)
    ]
    pd.DataFrame(portfolio_rows).to_csv(root / "portfolio.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        columns=["code", "name", "category", "score", "first_seen", "last_seen", "status"]
    ).to_csv(root / "candidate_watchlist.csv", index=False, encoding="utf-8")
    pd.DataFrame(universe_rows).to_csv(
        root / "investable_universe.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(columns=RUNTIME_CSV_SCHEMAS["paper_ledger_file"]).to_csv(
        root / "paper_trades.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(columns=RUNTIME_CSV_SCHEMAS["signal_history_file"]).to_csv(
        root / "signal_history.csv", index=False, encoding="utf-8"
    )

    profile = UserProfile(
        profile_name=profile_name,
        research_topics=DEFAULT_RESEARCH_TOPICS,
    )
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile_path


def clone_user_workspace(
    loaded: LoadedProfile,
    workspace: str | Path,
    force: bool = False,
) -> Path:
    root = _resolve_workspace(workspace)
    profile_path = root / "profile.json"
    if profile_path.exists() and not force:
        raise FileExistsError(f"Profile already exists: {profile_path}. Use --force to replace it.")
    root.mkdir(parents=True, exist_ok=True)

    file_fields = {
        "universe_file": "universe.csv",
        "portfolio_file": "portfolio.csv",
        "candidate_watchlist_file": "candidate_watchlist.csv",
        "investable_universe_file": "investable_universe.csv",
        "paper_ledger_file": "paper_trades.csv",
        "signal_history_file": "signal_history.csv",
    }
    for field_name, destination_name in file_fields.items():
        source = loaded.resolve(getattr(loaded.profile, field_name))
        destination = root / destination_name
        if source is not None and source.exists() and source.stat().st_size > 0:
            frame = pd.read_csv(source, dtype={"code": str})
            if field_name == "portfolio_file":
                frame = frame.reindex(columns=RUNTIME_CSV_SCHEMAS["portfolio_file"])
            frame.to_csv(destination, index=False, encoding="utf-8")
        else:
            columns = RUNTIME_CSV_SCHEMAS.get(field_name, [])
            pd.DataFrame(columns=columns).to_csv(destination, index=False, encoding="utf-8")

    profile = loaded.profile.model_copy(
        update={
            **file_fields,
            "output_root": "runs",
            "state_dir": ".agent",
        }
    )
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile_path


def validate_user_profile(loaded: LoadedProfile) -> list[str]:
    warnings: list[str] = []
    required_paths = {
        "universe_file": loaded.profile.universe_file,
        "portfolio_file": loaded.profile.portfolio_file,
    }
    for label, value in required_paths.items():
        path = loaded.resolve(value)
        if path is None or not path.exists():
            warnings.append(f"{label} does not exist: {path}")

    portfolio_path = loaded.resolve(loaded.profile.portfolio_file)
    if portfolio_path is not None and portfolio_path.exists():
        portfolio = pd.read_csv(portfolio_path, dtype={"code": str})
        if not portfolio.empty:
            weights = pd.to_numeric(
                portfolio.get("current_weight", pd.Series(dtype=float)), errors="coerce"
            )
            values = pd.to_numeric(
                portfolio.get("current_value_yuan", pd.Series(dtype=float)), errors="coerce"
            )
            if (weights.dropna() < 0).any():
                warnings.append("portfolio current_weight contains negative values.")
            if float(weights.fillna(0.0).sum()) > 1.000001:
                warnings.append("portfolio current_weight sums to more than 100%.")
            if (values.dropna() < 0).any():
                warnings.append("portfolio current_value_yuan contains negative values.")
            has_weights = float(weights.fillna(0.0).sum()) > 0
            has_values = float(values.fillna(0.0).sum()) > 0
            if not has_weights and not has_values:
                warnings.append("portfolio needs current_weight or current_value_yuan.")
    if not loaded.profile.human_approval_required:
        warnings.append("human_approval_required is disabled; this is not recommended.")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize and validate local user profiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--workspace", required=True)
    init_parser.add_argument("--name", default="local-user")
    init_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--profile", required=True)
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--profile", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        path = initialize_user_workspace(args.workspace, args.name, args.force)
        print(f"Initialized local profile: {path}")
        return 0
    loaded = load_user_profile(args.profile)
    if args.command == "show":
        print(json.dumps(loaded.profile.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    warnings = validate_user_profile(loaded)
    print(f"profile={loaded.path}")
    print(f"valid={not warnings}")
    for warning in warnings:
        print(f"warning={warning}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
