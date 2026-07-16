from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fund_agent.config import FundSpec
from fund_agent.return_forecast import ExpectedReturnForecast
from fund_agent.skills import (
    SkillDefinition,
    aggregate_skill_signal,
    run_fund_skills,
)
from fund_agent.user_profile import MarketActionSettings


@dataclass(frozen=True)
class FundAnalysisResult:
    table: pd.DataFrame
    skill_errors: tuple[str, ...]
    portfolio_summary: dict[str, Any]


def analyze_funds(
    prices: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    target_weights: pd.Series,
    portfolio: pd.DataFrame | None,
    forecast: ExpectedReturnForecast | None,
    skill_definitions: list[SkillDefinition],
    enabled_skill_ids: list[str],
    action_settings: MarketActionSettings | None = None,
) -> FundAnalysisResult:
    settings = action_settings or MarketActionSettings()
    facts = _build_facts(prices, fund_universe, forecast)
    skill_outputs, skill_errors = run_fund_skills(
        facts,
        skill_definitions,
        enabled_skill_ids,
    )
    current_weights, current_values, portfolio_value, has_portfolio = _current_portfolio(portfolio)
    name_by_code = {fund.code: fund.name for fund in fund_universe}
    category_by_code = {fund.code: fund.category for fund in fund_universe}
    rows = []
    for code, fact in facts.iterrows():
        current_weight = float(current_weights.get(code, 0.0))
        target_weight = float(target_weights.get(code, 0.0))
        signals = skill_outputs.get(str(code), [])
        skill_score, skill_confidence = aggregate_skill_signal(signals)
        decision_score, decision_confidence = _combine_decision_signal(
            technical_score=float(fact["technical_score"]),
            technical_confidence=float(fact["technical_confidence"]),
            context_score=_optional_number(fact.get("context_score")),
            context_confidence=float(fact.get("context_confidence", 0.0)),
            skill_score=skill_score,
            skill_confidence=skill_confidence,
        )
        signal = classify_research_signal(
            decision_score,
            decision_confidence,
            settings,
        )
        stance = (
            "市场偏强"
            if decision_score >= settings.decision_threshold
            else "市场偏弱"
            if decision_score <= -settings.decision_threshold
            else "市场中性"
        )
        reasons = [str(fact["technical_summary"])]
        if fact.get("context_confidence", 0.0) > 0:
            reasons.append(
                f"当日市场环境分数 {float(fact['context_score']):+.2f}，"
                f"置信度 {float(fact['context_confidence']):.0%}。"
            )
        reasons.extend(signal["summary"] for signal in signals if signal.get("summary"))
        risks = [signal["risk"] for signal in signals if signal.get("risk")]
        if fact.get("context_confidence", 0.0) < 0.25:
            risks.append("实时市场信号覆盖不足，本轮主要依赖历史净值技术面。")
        if current_weight <= 1e-8:
            risks.append("当前未记录持仓；该结论仅用于研究观察。")
        current_value = _optional_number(current_values.get(code))
        rows.append(
            {
                "code": code,
                "name": name_by_code.get(code, code),
                "category": category_by_code.get(code, "unknown"),
                "current_weight": current_weight if has_portfolio else np.nan,
                "current_value_yuan": current_value,
                "reference_target_weight": target_weight,
                "signal": signal,
                "stance": stance,
                "decision_score": decision_score,
                "decision_confidence": decision_confidence,
                "technical_score": fact["technical_score"],
                "technical_confidence": fact["technical_confidence"],
                "technical_summary": fact["technical_summary"],
                "ma5": fact["ma5"],
                "ma20": fact["ma20"],
                "ma60": fact["ma60"],
                "rsi14": fact["rsi14"],
                "macd_histogram": fact["macd_histogram"],
                "return_20d": fact["return_20d"],
                "return_60d": fact["return_60d"],
                "annual_volatility_60d": fact["annual_volatility_60d"],
                "max_drawdown_60d": fact["max_drawdown_60d"],
                "expected_return": fact["expected_return"],
                "context_score": fact["context_score"],
                "context_confidence": fact["context_confidence"],
                "skill_score": skill_score,
                "skill_confidence": skill_confidence,
                "skill_signals": signals,
                "reasons": list(dict.fromkeys(reasons))[:7],
                "risks": list(dict.fromkeys(risks))[:5],
            }
        )
    table = pd.DataFrame(rows).set_index("code")
    summary = {
        "fund_count": len(table),
        "strong_count": int((table["signal"] == "偏强").sum()),
        "weak_count": int((table["signal"] == "偏弱").sum()),
        "observe_count": int((table["signal"] == "观察").sum()),
        "risk_exit_count": int((table["signal"] == "风险退出").sum()),
        "portfolio_value_yuan": portfolio_value,
        "enabled_skill_count": len(enabled_skill_ids),
        "skill_error_count": len(skill_errors),
        "decision_basis": "nav_technical_and_market_context",
    }
    return FundAnalysisResult(table, tuple(skill_errors), summary)


def build_research_signals(table: pd.DataFrame) -> list[dict[str, Any]]:
    signals = []
    for code, row in table.iterrows():
        if row["signal"] == "观察":
            continue
        signal_code = {
            "偏强": "strong",
            "偏弱": "weak",
            "风险退出": "risk_exit",
        }[str(row["signal"])]
        signals.append(
            {
                "signal": signal_code,
                "signal_label": str(row["signal"]),
                "code": str(code),
                "name": str(row["name"]),
                "category": str(row["category"]),
                "signal_score": float(row["decision_score"]),
                "signal_confidence": float(row["decision_confidence"]),
                "expected_return": (
                    float(row["expected_return"])
                    if pd.notna(row["expected_return"])
                    else None
                ),
                "reason": str(row["technical_summary"]),
            }
        )
    return signals


def _build_facts(
    prices: pd.DataFrame,
    fund_universe: tuple[FundSpec, ...],
    forecast: ExpectedReturnForecast | None,
) -> pd.DataFrame:
    codes = [fund.code for fund in fund_universe if fund.code in prices.columns]
    returns = prices[codes].pct_change(fill_method=None)
    rows = []
    for code in codes:
        series = prices[code].dropna()
        recent_returns = returns[code].dropna().tail(60)
        equity = (1.0 + recent_returns).cumprod()
        max_drawdown = (
            float((equity / equity.cummax() - 1.0).min()) if not equity.empty else np.nan
        )
        technical = calculate_technical_signal(series)
        rows.append(
            {
                "code": code,
                "return_20d": _period_return(series, 20),
                "return_60d": _period_return(series, 60),
                "annual_volatility_60d": (
                    float(recent_returns.std(ddof=1) * np.sqrt(252))
                    if len(recent_returns) > 1
                    else np.nan
                ),
                "max_drawdown_60d": max_drawdown,
                **technical,
            }
        )
    facts = pd.DataFrame(rows).set_index("code")
    if forecast is None:
        facts["expected_return"] = np.nan
        facts["context_score"] = np.nan
        facts["context_confidence"] = 0.0
        return facts
    forecast_table = forecast.table.reindex(facts.index)
    facts["expected_return"] = forecast_table["blended_expected_return"]
    facts["context_score"] = forecast_table["context_score"]
    facts["context_confidence"] = forecast_table["context_confidence"].fillna(0.0)
    return facts


def calculate_technical_signal(series: pd.Series) -> dict[str, Any]:
    close = pd.to_numeric(series, errors="coerce").dropna()
    if len(close) < 20:
        return {
            "technical_score": 0.0,
            "technical_confidence": 0.0,
            "technical_summary": "净值历史不足 20 个交易日，无法形成技术面判断。",
            "ma5": np.nan,
            "ma20": np.nan,
            "ma60": np.nan,
            "rsi14": np.nan,
            "macd_histogram": np.nan,
        }

    latest = float(close.iloc[-1])
    ma5 = float(close.tail(5).mean())
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean()) if len(close) >= 60 else np.nan
    return_20d = _period_return(close, 20)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = float((macd - macd_signal).iloc[-1])
    daily_volatility = float(close.pct_change(fill_method=None).tail(20).std(ddof=1))
    macd_scale = max(abs(latest) * max(daily_volatility, 0.003), 1e-9)
    rsi14 = _rsi(close, 14)

    components: list[tuple[float, float]] = [
        (1.0 if ma5 >= ma20 else -1.0, 0.25),
        (1.0 if latest >= ma20 else -1.0, 0.10),
        (float(np.tanh(return_20d / 0.08)), 0.20),
        (float(np.tanh(macd_histogram / macd_scale)), 0.15),
    ]
    if pd.notna(ma60):
        components.append((1.0 if ma20 >= ma60 else -1.0, 0.20))
    if pd.notna(rsi14):
        components.append((float(np.clip((rsi14 - 50.0) / 25.0, -1.0, 1.0)), 0.10))
    available_weight = sum(weight for _, weight in components)
    score = sum(value * weight for value, weight in components) / available_weight
    ma60_text = f"MA60 {ma60:.4f}" if pd.notna(ma60) else "MA60 数据不足"
    rsi_text = f"RSI14 {rsi14:.1f}" if pd.notna(rsi14) else "RSI14 不可用"
    direction = "偏强" if score >= 0.20 else "偏弱" if score <= -0.20 else "震荡"
    return {
        "technical_score": float(np.clip(score, -1.0, 1.0)),
        "technical_confidence": float(np.clip(available_weight, 0.0, 1.0)),
        "technical_summary": (
            f"净值技术面{direction}：最新 {latest:.4f}，MA5 {ma5:.4f}，"
            f"MA20 {ma20:.4f}，{ma60_text}，20 日 {return_20d:+.1%}，{rsi_text}。"
        ),
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "rsi14": rsi14,
        "macd_histogram": macd_histogram,
    }


def _combine_decision_signal(
    technical_score: float,
    technical_confidence: float,
    context_score: float | None,
    context_confidence: float,
    skill_score: float,
    skill_confidence: float,
) -> tuple[float, float]:
    components = [(technical_score, technical_confidence, 0.60)]
    if context_score is not None:
        components.append((context_score, context_confidence, 0.25))
    if skill_confidence > 0:
        components.append((skill_score, skill_confidence, 0.15))
    effective_weight = sum(confidence * weight for _, confidence, weight in components)
    if effective_weight <= 0:
        return 0.0, 0.0
    score = sum(value * confidence * weight for value, confidence, weight in components)
    return (
        float(np.clip(score / effective_weight, -1.0, 1.0)),
        float(np.clip(effective_weight, 0.0, 1.0)),
    )


def classify_research_signal(
    score: float,
    confidence: float,
    settings: MarketActionSettings,
) -> str:
    if confidence < settings.minimum_confidence or abs(score) < settings.decision_threshold:
        return "观察"
    if score <= -settings.risk_exit_threshold:
        return "风险退出"
    return "偏强" if score > 0 else "偏弱"


def _current_portfolio(
    portfolio: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series, float, bool]:
    if portfolio is None or portfolio.empty or "code" not in portfolio:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0, False
    frame = portfolio.copy()
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    weights = pd.to_numeric(
        frame.get("current_weight", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    values = pd.to_numeric(
        frame.get("current_value_yuan", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    grouped_values = pd.Series(values.values, index=frame["code"]).groupby(level=0).sum(min_count=1)
    portfolio_value = float(values.fillna(0.0).sum())
    if float(weights.sum()) > 0:
        current = pd.Series(weights.values, index=frame["code"]).groupby(level=0).sum()
        return current, grouped_values, portfolio_value, True
    if portfolio_value > 0:
        current = pd.Series((values.fillna(0.0) / portfolio_value).values, index=frame["code"])
        return current.groupby(level=0).sum(), grouped_values, portfolio_value, True
    return pd.Series(dtype=float), grouped_values, 0.0, False


def _rsi(series: pd.Series, periods: int) -> float:
    delta = series.diff().dropna()
    if len(delta) < periods:
        return np.nan
    gain = float(delta.clip(lower=0.0).tail(periods).mean())
    loss = float((-delta.clip(upper=0.0)).tail(periods).mean())
    if gain == 0.0 and loss == 0.0:
        return 50.0
    if loss == 0.0:
        return 100.0
    relative_strength = gain / loss
    return float(100.0 - 100.0 / (1.0 + relative_strength))


def _optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _period_return(series: pd.Series, periods: int) -> float:
    if len(series) < 2:
        return np.nan
    start = max(0, len(series) - periods - 1)
    return float(series.iloc[-1] / series.iloc[start] - 1.0)
