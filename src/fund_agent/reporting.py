from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from fund_agent.agents import AgentBrief
from fund_agent.backtest import BacktestResult
from fund_agent.config import FundSpec, RiskProfile
from fund_agent.metrics import summarize_assets
from fund_agent.return_forecast import ExpectedReturnForecast


def write_markdown_report(
    output_path: str | Path,
    prices: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    profile: RiskProfile,
    briefs: list[AgentBrief],
    backtests: list[BacktestResult],
    target_weights: pd.Series,
    data_source: str,
    rebalance_freq: str,
    lookback_days: int,
    chart_paths: dict[str, Path] | None = None,
    forecast: ExpectedReturnForecast | None = None,
    target_strategy: str = "historical_max_sharpe",
    fund_analyses: pd.DataFrame | None = None,
    portfolio_summary: dict[str, object] | None = None,
    skill_errors: tuple[str, ...] | list[str] = (),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_summary = summarize_assets(prices)
    name_map = {fund.code: fund.name for fund in fund_universe}

    lines: list[str] = []
    lines.append("# 基金组合研究 Agent 报告")
    lines.append("")
    lines.append("本报告仅用于研究和学习，不构成投资建议。")
    lines.append("")
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- 数据源：`{data_source}`")
    lines.append(f"- 风险偏好：`{profile.name.value}`")
    lines.append(f"- 回测再平衡频率：`{rebalance_freq}`")
    lines.append(f"- 历史窗口：`{lookback_days}` 个交易日")
    lines.append("- 日频动作依据：`净值技术面 + 当日市场环境 + Skill 信号`")
    lines.append(f"- 组合优化参考：`{target_strategy}`（仅用于回测对照）")
    lines.append(f"- 权益类仓位上限：`{profile.max_equity_weight:.0%}`")
    lines.append(f"- 单只基金权重上限：`{profile.max_single_asset_weight:.0%}`")
    lines.append(f"- 回测单次换手上限：`{profile.max_turnover:.0%}`")
    fee_models = sorted({result.fee_model for result in backtests})
    lines.append(f"- 回测费用模型：`{', '.join(fee_models)}`")
    lines.append("")

    lines.append("## Agent 摘要")
    lines.append("")
    for brief in briefs:
        lines.append(f"### {brief.role}")
        lines.append("")
        for finding in brief.findings:
            lines.append(f"- {finding}")
        lines.append("")

    if forecast is not None:
        lines.append("## 离线上下文研究特征")
        lines.append("")
        lines.append(f"- 上下文来源：`{forecast.context_source}`")
        lines.append(f"- 市场数据日期：`{forecast.context_market_data_as_of or '未知'}`")
        lines.append(f"- 生成或加载时间：`{forecast.context_fetched_at}`")
        lines.append("- 这些估计只作为研究模型输入，未混入下方历史回测，也不是未来收益预测。")
        lines.append("")
        lines.append(
            "| 代码 | 基金 | 历史年化估计 | 短期代理 | 类别代理 | 新闻特征 | 政策特征 | 置信度 | 上下文修正 | 融合估计 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for code, row in forecast.table.iterrows():
            lines.append(
                f"| {code} | {name_map.get(code, '')} | {_pct(row['historical_annual_return'])} | "
                f"{_pct(row['intraday_change'])} | {_pct(row['sector_change'])} | "
                f"{_number(row['news_sentiment'])} | {_number(row['policy_sentiment'])} | "
                f"{_pct(row['context_confidence'])} | {_pct(row['context_annual_tilt'])} | "
                f"{_pct(row['blended_expected_return'])} |"
            )
        lines.append("")
        for note in forecast.notes:
            lines.append(f"- {note}")
        lines.append("")
        evidence_rows = forecast.table[forecast.table["news_evidence"].astype(str).str.len() > 0]
        if not evidence_rows.empty:
            lines.append("### 新闻与政策证据")
            lines.append("")
            for code, row in evidence_rows.iterrows():
                lines.append(f"- `{code}` {name_map.get(code, '')}：{row['news_evidence']}")
            lines.append("")

    if fund_analyses is not None and not fund_analyses.empty:
        _append_fund_analysis(
            lines,
            fund_analyses,
            portfolio_summary or {},
            skill_errors,
        )

    lines.append("## 组合优化参考权重")
    lines.append("")
    lines.append("该权重只用于组合研究与回测对照，不生成交易动作或金额。")
    lines.append("")
    if chart_paths and "target_weights" in chart_paths:
        lines.append(f"![Target Weights]({_relative_link(output, chart_paths['target_weights'])})")
        lines.append("")
    lines.append("| 代码 | 基金 | 目标权重 |")
    lines.append("|---|---|---:|")
    for code, weight in target_weights.sort_values(ascending=False).items():
        lines.append(f"| {code} | {name_map.get(code, '')} | {weight:.2%} |")
    lines.append("")

    lines.append("## 基金风险摘要")
    lines.append("")
    lines.append("| 代码 | 基金 | 年化收益 | 年化波动 | 最大回撤 | 夏普 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for code, row in asset_summary.iterrows():
        lines.append(
            f"| {code} | {name_map.get(code, '')} | {row['annual_return']:.2%} | "
            f"{row['annual_volatility']:.2%} | {row['max_drawdown']:.2%} | {row['sharpe']:.2f} |"
        )
    lines.append("")

    lines.append("## Walk-forward 回测")
    lines.append("")
    if chart_paths and "equity_curve" in chart_paths:
        lines.append(f"![Equity Curve]({_relative_link(output, chart_paths['equity_curve'])})")
        lines.append("")
    if chart_paths and "drawdown_curve" in chart_paths:
        lines.append(f"![Drawdown Curve]({_relative_link(output, chart_paths['drawdown_curve'])})")
        lines.append("")
    lines.append("| 策略 | 累计收益 | 年化收益 | 年化波动 | 最大回撤 | 夏普 | 平均换手 | 受限换手 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in backtests:
        s = result.summary
        lines.append(
            f"| {result.strategy} | {s['cumulative_return']:.2%} | {s['annual_return']:.2%} | "
            f"{s['annual_volatility']:.2%} | {s['max_drawdown']:.2%} | {s['sharpe']:.2f} | "
            f"{s['average_turnover']:.2%} | {s['blocked_turnover_total']:.2%} |"
        )
    lines.append("")

    best = max(backtests, key=lambda item: item.summary["sharpe"])
    lowest_drawdown = max(backtests, key=lambda item: item.summary["max_drawdown"])
    lines.append("## 复盘摘要")
    lines.append("")
    lines.append(f"- 本次历史回测最高夏普：`{best.strategy}`，夏普 `{best.summary['sharpe']:.2f}`。")
    lines.append(
        f"- 本次历史回测最低最大回撤：`{lowest_drawdown.strategy}`，"
        f"最大回撤 `{lowest_drawdown.summary['max_drawdown']:.2%}`。"
    )
    lines.append("- 最新上下文目标与历史策略回测口径不同，不能直接比较或宣称带来收益提升。")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _relative_link(report_path: Path, asset_path: Path) -> str:
    return Path(os.path.relpath(asset_path, start=report_path.parent)).as_posix()


def _pct(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):.2%}"


def _number(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):+.2f}"


def _plain_number(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):.1f}"


def _append_fund_analysis(
    lines: list[str],
    analyses: pd.DataFrame,
    summary: dict[str, object],
    skill_errors: tuple[str, ...] | list[str],
) -> None:
    lines.extend(
        [
            "## 逐基金净值技术面与市场环境判断",
            "",
            (
                f"- 本轮逐一分析 `{int(summary.get('fund_count', len(analyses)))}` 只基金；"
                f"偏强 `{int(summary.get('strong_count', 0))}` 只，"
                f"偏弱 `{int(summary.get('weak_count', 0))}` 只，"
                f"观察 `{int(summary.get('observe_count', 0))}` 只，"
                f"风险退出 `{int(summary.get('risk_exit_count', 0))}` 只。"
            ),
            "- 信号由每只基金的净值技术面、离线市场上下文和 Skill 共同决定。",
            "- 公开版不生成开仓、加仓、减仓或金额建议；组合优化目标仅作研究参考。",
            "- 研究信号尚未纳入下方 walk-forward 回测，不能用组合策略回测证明其有效。",
            "",
            "| 代码 | 基金 | 信号 | 研判 | 当前权重 | 技术分 | 市场分 | 决策分 | 置信度 | RSI14 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code, row in analyses.iterrows():
        lines.append(
            f"| {code} | {row['name']} | {row['signal']} | {row['stance']} | "
            f"{_pct(row['current_weight'])} | "
            f"{_number(row['technical_score'])} | {_number(row['context_score'])} | "
            f"{_number(row['decision_score'])} | {_pct(row['decision_confidence'])} | "
            f"{_plain_number(row['rsi14'])} |"
        )
    lines.append("")
    for code, row in analyses.iterrows():
        lines.extend([f"### `{code}` {row['name']}", ""])
        reasons = row.get("reasons", [])
        risks = row.get("risks", [])
        lines.append(f"- 研究信号：**{row['signal']}**，当前研判为 **{row['stance']}**。")
        for reason in reasons if isinstance(reasons, list) else []:
            lines.append(f"- 依据：{reason}")
        for risk in risks if isinstance(risks, list) else []:
            lines.append(f"- 风险：{risk}")
        lines.append("")
    if skill_errors:
        lines.extend(["### Skill 运行异常", ""])
        for error in skill_errors:
            lines.append(f"- `{error}`")
        lines.append("")
