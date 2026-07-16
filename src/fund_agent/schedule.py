from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.user_profile import LoadedProfile, UserProfile


@dataclass(frozen=True)
class ScheduledTask:
    task_type: str
    scheduled_at: datetime
    reason: str


def load_trade_dates(
    loaded: LoadedProfile,
    refresh: bool = False,
) -> tuple[set[date], str]:
    state_dir = loaded.resolve(loaded.profile.state_dir)
    if state_dir is None:
        raise ValueError("state_dir is required")
    cache_path = state_dir / "trading_calendar.csv"
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path)
        dates = _trade_dates_from_frame(frame)
        if dates:
            return dates, "cache"

    if loaded.profile.calendar_source == "weekdays":
        today = datetime.now(ZoneInfo(loaded.profile.timezone)).date()
        start = today - timedelta(days=370)
        end = today + timedelta(days=370)
        dates = {timestamp.date() for timestamp in pd.bdate_range(start, end)}
        return dates, "weekday fallback configured by user"

    today = datetime.now(ZoneInfo(loaded.profile.timezone)).date()
    start = today - timedelta(days=370)
    end = today + timedelta(days=370)
    calendar_loader = require_adapter("trading_calendar", get_adapters().trading_calendar)
    dates = calendar_loader(start, end)
    if not dates:
        raise RuntimeError("Trading calendar returned no dates.")
    state_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": sorted(dates)}).to_csv(cache_path, index=False, encoding="utf-8")
    return dates, "plugin"


def due_tasks(
    now: datetime,
    profile: UserProfile,
    trade_dates: set[date],
    completed: set[tuple[str, date]] | None = None,
) -> list[ScheduledTask]:
    zone = ZoneInfo(profile.timezone)
    localized = now.astimezone(zone) if now.tzinfo else now.replace(tzinfo=zone)
    today = localized.date()
    completed = completed or set()
    if today not in trade_dates:
        return []

    tasks: list[ScheduledTask] = []
    daily_at = datetime.combine(today, _parse_time(profile.schedule.daily_time), zone)
    if localized >= daily_at and ("daily_rebalance", today) not in completed:
        tasks.append(ScheduledTask("daily_rebalance", daily_at, "交易日基金池离线研究信号"))

    weekly_at = datetime.combine(today, _parse_time(profile.schedule.weekly_time), zone)
    if today.weekday() == 4 and localized >= weekly_at and ("weekly_screen", today) not in completed:
        tasks.append(ScheduledTask("weekly_screen", weekly_at, "周五离线候选样例筛选"))

    monthly_date = last_trading_day_of_last_week(today.year, today.month, trade_dates)
    monthly_at = datetime.combine(today, _parse_time(profile.schedule.monthly_time), zone)
    if (
        today == monthly_date
        and localized >= monthly_at
        and ("monthly_research", today) not in completed
    ):
        tasks.append(ScheduledTask("monthly_research", monthly_at, "月末最后交易周离线主题研究"))
    return tasks


def last_trading_day_of_last_week(
    year: int,
    month: int,
    trade_dates: set[date],
) -> date | None:
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    last_day = next_month - timedelta(days=1)
    friday = last_day - timedelta(days=(last_day.weekday() - 4) % 7)
    last_week_monday = friday - timedelta(days=4)
    week_dates = sorted(
        day
        for day in trade_dates
        if last_week_monday <= day <= friday
    )
    if week_dates:
        return week_dates[-1]
    month_dates = sorted(day for day in trade_dates if day.year == year and day.month == month)
    return month_dates[-1] if month_dates else None


def _trade_dates_from_frame(frame: pd.DataFrame) -> set[date]:
    if frame.empty:
        return set()
    candidates = [column for column in frame.columns if "date" in str(column).lower() or "日期" in str(column)]
    column = candidates[0] if candidates else frame.columns[0]
    parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
    return {timestamp.date() for timestamp in parsed}


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()
