from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from fund_agent.schedule import due_tasks, last_trading_day_of_last_week
from fund_agent.user_profile import UserProfile


def test_friday_tasks_and_monthly_research_times() -> None:
    profile = UserProfile()
    trade_dates = {timestamp.date() for timestamp in pd.bdate_range("2026-06-01", "2026-06-30")}
    zone = ZoneInfo("Asia/Shanghai")

    afternoon = due_tasks(datetime(2026, 6, 26, 14, 30, tzinfo=zone), profile, trade_dates)
    evening = due_tasks(datetime(2026, 6, 26, 20, 0, tzinfo=zone), profile, trade_dates)

    assert {task.task_type for task in afternoon} == {"daily_rebalance", "weekly_screen"}
    assert {task.task_type for task in evening} == {
        "daily_rebalance",
        "weekly_screen",
        "monthly_research",
    }
    assert last_trading_day_of_last_week(2026, 6, trade_dates) == date(2026, 6, 26)


def test_non_trading_day_has_no_tasks_and_completed_tasks_are_skipped() -> None:
    profile = UserProfile()
    trade_dates = {date(2026, 6, 26)}
    zone = ZoneInfo("Asia/Shanghai")

    saturday = due_tasks(datetime(2026, 6, 27, 20, 0, tzinfo=zone), profile, trade_dates)
    friday = due_tasks(
        datetime(2026, 6, 26, 20, 0, tzinfo=zone),
        profile,
        trade_dates,
        completed={
            ("daily_rebalance", date(2026, 6, 26)),
            ("weekly_screen", date(2026, 6, 26)),
            ("monthly_research", date(2026, 6, 26)),
        },
    )

    assert not saturday
    assert not friday
