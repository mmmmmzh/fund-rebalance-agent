from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.config import FundSpec
from fund_agent.user_profile import UserProfile


BoardLoader = Callable[[str, str, date], pd.DataFrame]


def collect_monthly_research(
    profile: UserProfile,
    fund_universe: tuple[FundSpec, ...],
    output_path: str | Path,
    as_of: date | None = None,
    board_loader: BoardLoader | None = None,
    **_: object,
) -> dict[str, object]:
    as_of = as_of or date.today()
    loader = board_loader or _synthetic_board_loader
    evidence: list[dict[str, object]] = []
    if profile.data_source == "plugin":
        research_loader = require_adapter("research", get_adapters().research)
        evidence = research_loader(profile.research_topics, fund_universe, as_of)

    rows = []
    for topic in profile.research_topics:
        series = loader(topic.board_type, topic.board_name, as_of)
        metrics = board_trend_metrics(series)
        rows.append(
            {
                "topic": topic.name,
                "board": topic.board_name,
                **metrics,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_report(output, rows, evidence, as_of)
    return {
        "report_path": str(output),
        "topic_count": len(rows),
        "evidence_count": len(evidence),
        "rag_topic_count": 0,
    }


def board_trend_metrics(frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty:
        return {"return_20d": np.nan, "return_60d": np.nan, "volatility_20d": np.nan, "trend": "数据不足"}
    close_column = "close" if "close" in frame else frame.columns[-1]
    close = pd.to_numeric(frame[close_column], errors="coerce").dropna()
    if len(close) < 2:
        return {"return_20d": np.nan, "return_60d": np.nan, "volatility_20d": np.nan, "trend": "数据不足"}
    daily = close.pct_change(fill_method=None).dropna()
    return_20d = float(close.iloc[-1] / close.iloc[max(0, len(close) - 21)] - 1.0)
    return_60d = float(close.iloc[-1] / close.iloc[max(0, len(close) - 61)] - 1.0)
    volatility = float(daily.tail(20).std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else np.nan
    trend = "偏强" if return_20d > 0.03 else "偏弱" if return_20d < -0.03 else "观察"
    return {
        "return_20d": return_20d,
        "return_60d": return_60d,
        "volatility_20d": volatility,
        "trend": trend,
    }


def _synthetic_board_loader(board_type: str, board_name: str, as_of: date) -> pd.DataFrame:
    seed = sum(ord(character) for character in f"{board_type}:{board_name}") % (2**32)
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=as_of, periods=90)
    drift = (seed % 7 - 3) / 100000
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, len(index)))
    return pd.DataFrame({"date": index, "close": close}).set_index("date")


def _write_report(
    path: Path,
    rows: list[dict[str, object]],
    evidence: list[dict[str, object]],
    as_of: date,
) -> None:
    lines = [
        "# 月度离线主题研究",
        "",
        f"- 研究日期：`{as_of.isoformat()}`",
        "- 默认趋势序列为确定性合成数据，仅用于演示 Agent 工作流。",
        "- 公开版不下载外部报告、不执行 PDF RAG，也不调用外部模型。",
        "",
        "| 主题 | 序列 | 20 日 | 60 日 | 年化波动 | 信号 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['topic']} | {row['board']} | {_pct(row['return_20d'])} | "
            f"{_pct(row['return_60d'])} | {_pct(row['volatility_20d'])} | {row['trend']} |"
        )
    if not rows:
        lines.append("| - | - | - | - | - | 未配置研究主题 |")
    if evidence:
        lines.extend(["", "## 插件证据", ""])
        for item in evidence[:20]:
            lines.append(f"- {str(item.get('title', '未命名证据'))[:200]}")
    lines.extend(["", "> 所有结论均需人工核验，不构成投资建议。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"
