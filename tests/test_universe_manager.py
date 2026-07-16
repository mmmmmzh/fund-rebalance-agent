from __future__ import annotations

from fund_agent.universe_manager import add_fund, remove_fund


def test_add_and_remove_fund(tmp_path) -> None:
    path = tmp_path / "universe.csv"

    add_fund(
        path,
        code="123",
        name="测试基金C",
        category="test",
        is_equity_like=True,
    )

    text = path.read_text(encoding="utf-8")
    assert "000123" in text
    assert "测试基金C" in text

    assert remove_fund(path, "000123") is True
    assert remove_fund(path, "000123") is False
