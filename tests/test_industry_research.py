from __future__ import annotations

from datetime import date

import pandas as pd

from fund_agent.config import DEFAULT_FUND_UNIVERSE
from fund_agent.industry_research import board_trend_metrics, collect_monthly_research
from fund_agent.user_profile import ResearchTopic, UserProfile


def test_board_trend_metrics_detects_positive_trend() -> None:
    dates = pd.bdate_range("2026-01-01", periods=80)
    metrics = board_trend_metrics(pd.DataFrame({"close": range(100, 180)}, index=dates))

    assert metrics["return_20d"] > 0
    assert metrics["return_60d"] > 0
    assert metrics["trend"] == "偏强"


def test_monthly_research_writes_offline_topic_report(tmp_path) -> None:
    profile = UserProfile(
        research_topics=[ResearchTopic(name="成长", board_type="concept", board_name="growth")]
    )
    dates = pd.bdate_range("2026-01-01", periods=100)
    output = tmp_path / "monthly.md"

    result = collect_monthly_research(
        profile,
        DEFAULT_FUND_UNIVERSE,
        output,
        as_of=date(2026, 6, 26),
        board_loader=lambda *_: pd.DataFrame({"close": range(100, 200)}, index=dates),
    )

    assert result["topic_count"] == 1
    assert result["rag_topic_count"] == 0
    text = output.read_text(encoding="utf-8")
    assert "月度离线主题研究" in text
    assert "不下载外部报告" in text
