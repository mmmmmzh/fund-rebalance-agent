from __future__ import annotations

import pandas as pd
import pytest

from fund_agent.fund_screener import screen_fund_candidates, write_screening_report


def test_local_candidate_screen_ranks_and_limits_categories(tmp_path) -> None:
    candidates = pd.DataFrame(
        [
            {"code": "1", "name": "A", "category": "equity", "candidate_score": 0.9},
            {"code": "2", "name": "B", "category": "equity", "candidate_score": 0.8},
            {"code": "3", "name": "C", "category": "bond", "candidate_score": 0.7},
        ]
    )

    result = screen_fund_candidates(candidates, top_n=3, max_per_category=1)
    report = write_screening_report(tmp_path / "screen.md", result)

    assert result["code"].tolist() == ["000001", "000003"]
    assert result["candidate_score"].is_monotonic_decreasing
    text = report.read_text(encoding="utf-8")
    assert "不代表全市场覆盖" in text
    assert "不构成基金推荐" in text


def test_local_candidate_screen_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        screen_fund_candidates(pd.DataFrame({"code": ["1"]}))
