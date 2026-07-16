from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
import json
import operator
from pathlib import Path
import re
from typing import Annotated, Any, Iterator
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from fund_agent.agents import market_analyst_brief
from fund_agent.config import FundSpec, get_risk_profile
from fund_agent.data import PriceDataset, load_price_dataset
from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.fund_screener import write_screening_report
from fund_agent.fund_analysis import analyze_funds, build_research_signals
from fund_agent.industry_research import collect_monthly_research
from fund_agent.market_context import MarketContextSnapshot, load_market_context
from fund_agent.paper_trading import record_run_recommendations
from fund_agent.pipeline import backtest_summary_frame, run_analysis
from fund_agent.reporting import write_markdown_report
from fund_agent.skills import load_skill_definitions
from fund_agent.universe import load_fund_universe_csv
from fund_agent.user_profile import LoadedProfile, ensure_user_runtime_files, load_user_profile
from fund_agent.visualization import generate_report_charts


class AgentState(TypedDict, total=False):
    run_id: str
    task_type: str
    profile_path: str
    scheduled_for: str
    run_dir: str
    status: str
    effective_lookback: int
    universe_codes: list[str]
    prices_path: str
    context_path: str | None
    context_meta_path: str | None
    data_source: str
    market_findings: list[str]
    candidate_report_path: str
    candidate_csv_path: str
    candidate_codes: list[str]
    monthly_report_path: str
    target_weights: dict[str, float]
    backtest_summary: list[dict[str, Any]]
    analysis_report_path: str
    research_actions: list[dict[str, Any]]
    proposed_actions: list[dict[str, Any]]
    fund_analyses: list[dict[str, Any]]
    portfolio_summary: dict[str, Any]
    skill_errors: list[str]
    risk_checks: dict[str, Any]
    approval_required: bool
    approval_payload: dict[str, Any]
    approval: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]


def build_agent_graph(checkpointer):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install agent dependencies with: pip install -e .[agent]") from exc

    builder = StateGraph(AgentState)
    builder.add_node("profile_agent", profile_agent)
    builder.add_node("data_collection_agent", data_collection_agent)
    builder.add_node("market_research_agent", market_research_agent)
    builder.add_node("portfolio_agent", portfolio_agent)
    builder.add_node("fund_screening_agent", fund_screening_agent)
    builder.add_node("monthly_research_agent", monthly_research_agent)
    builder.add_node("risk_fee_agent", risk_fee_agent)
    builder.add_node("review_agent", review_agent)
    builder.add_node("human_approval_agent", human_approval_agent)
    builder.add_node("finalize_agent", finalize_agent)

    builder.add_edge(START, "profile_agent")
    builder.add_conditional_edges(
        "profile_agent",
        lambda state: state["task_type"],
        {
            "daily_rebalance": "data_collection_agent",
            "weekly_screen": "fund_screening_agent",
            "monthly_research": "monthly_research_agent",
        },
    )
    builder.add_edge("data_collection_agent", "market_research_agent")
    builder.add_edge("market_research_agent", "portfolio_agent")
    builder.add_edge("portfolio_agent", "risk_fee_agent")
    builder.add_edge("fund_screening_agent", "risk_fee_agent")
    builder.add_edge("risk_fee_agent", "review_agent")
    builder.add_edge("monthly_research_agent", "review_agent")
    builder.add_conditional_edges(
        "review_agent",
        lambda state: "approval" if state.get("approval_required") else "complete",
        {"approval": "human_approval_agent", "complete": "finalize_agent"},
    )
    builder.add_edge("human_approval_agent", "finalize_agent")
    builder.add_edge("finalize_agent", END)
    return builder.compile(checkpointer=checkpointer)


def run_agent_workflow(
    profile_path: str | Path,
    task_type: str,
    scheduled_for: datetime | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    loaded = load_user_profile(profile_path)
    if task_type not in {"daily_rebalance", "weekly_screen", "monthly_research"}:
        raise ValueError(f"Unsupported Agent task: {task_type}")
    scheduled_for = scheduled_for or datetime.now(ZoneInfo(loaded.profile.timezone))
    run_id = run_id or _new_run_id(task_type, scheduled_for)
    run_dir = _run_directory(loaded, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    initial: AgentState = {
        "run_id": run_id,
        "task_type": task_type,
        "profile_path": str(loaded.path),
        "scheduled_for": scheduled_for.isoformat(),
        "run_dir": str(run_dir),
        "status": "running",
        "trace": [],
    }
    with _compiled_graph(loaded) as graph:
        result = graph.invoke(initial, config=_thread_config(run_id))
    return _normalize_graph_result(result)


def resume_agent_workflow(
    profile_path: str | Path,
    run_id: str,
    decision: str,
    feedback: str = "",
) -> dict[str, Any]:
    loaded = load_user_profile(profile_path)
    try:
        from langgraph.types import Command
    except ImportError as exc:
        raise RuntimeError("Install agent dependencies with: pip install -e .[agent]") from exc
    payload = {
        "decision": decision,
        "feedback": feedback,
        "reviewed_at": datetime.now(ZoneInfo(loaded.profile.timezone)).isoformat(),
    }
    with _compiled_graph(loaded) as graph:
        result = graph.invoke(Command(resume=payload), config=_thread_config(run_id))
    return _normalize_graph_result(result)


def workflow_status(profile_path: str | Path, run_id: str) -> dict[str, Any]:
    loaded = load_user_profile(profile_path)
    with _compiled_graph(loaded) as graph:
        snapshot = graph.get_state(_thread_config(run_id))
    values = dict(snapshot.values) if snapshot.values else {}
    values["next_nodes"] = list(snapshot.next)
    return values


def profile_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    _ensure_runtime_files(loaded)
    return {
        "status": "profile_loaded",
        "trace": [_event("ProfileAgent", f"已加载本地用户配置 {loaded.profile.profile_name}")],
    }


def data_collection_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    universe = _effective_universe(loaded)
    profile = loaded.profile
    scheduled_date = _scheduled_datetime(state).date().isoformat()
    dataset = load_price_dataset(
        source=profile.data_source,
        start=profile.start_date,
        end=scheduled_date,
        fund_universe=universe,
    )
    effective_lookback = _safe_lookback(dataset.prices, profile.lookback_days)
    if effective_lookback is None:
        raise ValueError(f"NAV history is too short: rows={len(dataset.prices)}")

    run_dir = Path(state["run_dir"])
    prices_path = run_dir / "prices.csv"
    dataset.prices.to_csv(prices_path, index_label="date", encoding="utf-8")

    context = load_market_context(
        profile.market_context_source,
        universe,
        dataset.prices,
    )
    context_path = None
    context_meta_path = None
    if context is not None:
        context_path = run_dir / "market_context.csv"
        context_meta_path = run_dir / "market_context_meta.json"
        context.signals.to_csv(context_path, index_label="code", encoding="utf-8")
        context_meta_path.write_text(
            json.dumps(
                {
                    "source": context.source,
                    "fetched_at": context.fetched_at.isoformat(),
                    "market_data_as_of": context.market_data_as_of,
                    "notes": list(context.notes),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "status": "data_collected",
        "effective_lookback": effective_lookback,
        "universe_codes": [fund.code for fund in universe],
        "prices_path": str(prices_path),
        "context_path": str(context_path) if context_path else None,
        "context_meta_path": str(context_meta_path) if context_meta_path else None,
        "data_source": dataset.source,
        "trace": [
            _event(
                "DataCollectionAgent",
                f"加载 {len(universe)} 只研究标的、{len(dataset.prices)} 行价格及市场上下文",
            )
        ],
    }


def market_research_agent(state: AgentState) -> dict[str, Any]:
    prices = _read_prices(state["prices_path"])
    brief = market_analyst_brief(prices)
    findings = list(brief.findings)
    context = _read_context(state)
    if context is not None:
        components = context.signals[
            ["intraday_change", "sector_change", "news_sentiment", "policy_sentiment"]
        ]
        findings.append(f"Current-context field coverage: {components.notna().mean().mean():.0%}.")
        findings.append(f"Context source: {context.source}.")
    return {
        "status": "market_researched",
        "market_findings": findings,
        "trace": [_event("MarketResearchAgent", "完成离线市场状态与可用上下文特征研究")],
    }


def portfolio_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    universe = _effective_universe(loaded)
    prices = _read_prices(state["prices_path"])
    dataset = PriceDataset(prices=prices, fund_universe=universe, source=state["data_source"])
    context = _read_context(state)
    profile = loaded.profile
    analysis = run_analysis(
        dataset,
        get_risk_profile(profile.risk_profile),
        state["effective_lookback"],
        profile.rebalance_freq,
        market_context=context,
    )
    run_dir = Path(state["run_dir"])
    report_path = run_dir / "daily_rebalance_report.md"
    chart_paths = generate_report_charts(
        run_dir / "assets",
        analysis.backtests,
        analysis.target_weights,
    )
    portfolio_path = loaded.resolve(profile.portfolio_file)
    portfolio = (
        pd.read_csv(portfolio_path, dtype={"code": str})
        if portfolio_path is not None and portfolio_path.exists()
        else pd.DataFrame()
    )
    portfolio_error = _portfolio_input_error(portfolio)
    if portfolio_error:
        raise ValueError(portfolio_error)
    skills_dir = loaded.resolve(profile.skills_dir)
    skill_definitions = load_skill_definitions(skills_dir)
    fund_analysis = analyze_funds(
        prices=prices,
        fund_universe=universe,
        target_weights=analysis.target_weights,
        portfolio=portfolio,
        forecast=analysis.forecast,
        skill_definitions=skill_definitions,
        enabled_skill_ids=profile.enabled_skills,
        action_settings=profile.market_actions,
    )
    write_markdown_report(
        report_path,
        prices,
        universe,
        get_risk_profile(profile.risk_profile),
        analysis.briefs,
        analysis.backtests,
        analysis.target_weights,
        dataset.source,
        profile.rebalance_freq,
        state["effective_lookback"],
        chart_paths,
        forecast=analysis.forecast,
        target_strategy=analysis.target_strategy,
        fund_analyses=fund_analysis.table,
        portfolio_summary=fund_analysis.portfolio_summary,
        skill_errors=fund_analysis.skill_errors,
    )
    signals = build_research_signals(fund_analysis.table)
    fund_records = _serializable_records(fund_analysis.table)
    return {
        "status": "portfolio_proposed",
        "target_weights": {code: float(weight) for code, weight in analysis.target_weights.items()},
        "backtest_summary": backtest_summary_frame(analysis.backtests).to_dict(orient="records"),
        "analysis_report_path": str(report_path),
        "research_actions": signals,
        "proposed_actions": signals,
        "fund_analyses": fund_records,
        "portfolio_summary": fund_analysis.portfolio_summary,
        "skill_errors": list(fund_analysis.skill_errors),
        "trace": [
            _event(
                "PortfolioAgent",
                f"逐一分析 {len(fund_records)} 只基金的净值技术面与市场环境，"
                f"生成 {len(signals)} 项需复核研究信号",
            )
        ],
    }


def fund_screening_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    profile = loaded.profile
    settings = profile.screening
    if profile.data_source == "sample":
        candidates = _sample_screen_candidates(settings.candidate_top_n * 2)
    else:
        loader = require_adapter("candidates", get_adapters().candidates)
        candidates = loader(max(settings.candidate_top_n * 2, 8))
    candidates = candidates.head(settings.candidate_top_n).copy()
    run_dir = Path(state["run_dir"])
    report_path = run_dir / "weekly_candidate_screen.md"
    csv_path = run_dir / "weekly_candidate_screen.csv"
    write_screening_report(report_path, candidates)
    candidates.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _update_candidate_watchlist(loaded, candidates, _scheduled_datetime(state).date())

    existing = {fund.code for fund in _effective_universe(loaded)}
    new_candidates = candidates[~candidates["code"].astype(str).str.zfill(6).isin(existing)]
    new_candidates = new_candidates.head(settings.max_new_funds)
    actions = [
        {
            "operation": "add_to_research_universe",
            "code": str(row["code"]).zfill(6),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "score": float(row.get("deep_score", row["candidate_score"])),
            "research_only": True,
        }
        for _, row in new_candidates.iterrows()
    ]
    return {
        "status": "candidates_screened",
        "candidate_report_path": str(report_path),
        "candidate_csv_path": str(csv_path),
        "candidate_codes": [str(code).zfill(6) for code in candidates["code"]],
        "proposed_actions": actions,
        "trace": [
            _event(
                "FundScreeningAgent",
                f"完成离线候选样例排序；更新观察池 Top {len(candidates)}，"
                f"提出 {len(actions)} 只待审批研究标的",
            )
        ],
    }


def monthly_research_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    universe = _effective_universe(loaded)
    output_path = Path(state["run_dir"]) / "monthly_industry_research.md"
    result = collect_monthly_research(
        loaded.profile,
        universe,
        output_path,
        as_of=_scheduled_datetime(state).date(),
    )
    return {
        "status": "monthly_research_completed",
        "monthly_report_path": result["report_path"],
        "proposed_actions": [],
        "trace": [
            _event(
                "IndustryResearchAgent",
                f"完成 {result['topic_count']} 个离线主题研究；未执行外部报告检索",
            )
        ],
    }


def risk_fee_agent(state: AgentState) -> dict[str, Any]:
    loaded = load_user_profile(state["profile_path"])
    research_actions = [
        dict(action)
        for action in state.get("research_actions", state.get("proposed_actions", []))
    ]
    actions = [dict(action) for action in research_actions]
    checks: dict[str, Any] = {"hard_violations": [], "warnings": []}
    if state["task_type"] == "daily_rebalance":
        universe = _effective_universe(loaded)
        profile = get_risk_profile(loaded.profile.risk_profile)
        current = _load_current_weights(loaded)
        equity_codes = [fund.code for fund in universe if fund.is_equity_like]
        current_equity_weight = float(current.reindex(equity_codes).fillna(0.0).sum())
        checks.update(
            {
                "decision_basis": "offline_nav_technical_and_plugin_signals",
                "current_equity_weight": current_equity_weight,
                "current_total_weight": float(current.sum()),
                "max_single_weight": float(current.max()) if not current.empty else 0.0,
                "equity_cap": profile.max_equity_weight,
                "single_asset_cap": profile.max_single_asset_weight,
                "review_signal_count": len(actions),
                "risk_exit_count": sum(action.get("signal") == "risk_exit" for action in actions),
            }
        )
        if current_equity_weight > profile.max_equity_weight + 1e-6:
            checks["warnings"].append("当前权益类暴露高于所选研究风险上限")
        if not current.empty and float(current.max()) > profile.max_single_asset_weight + 1e-6:
            checks["warnings"].append("当前单一基金权重高于所选研究风险上限")
        if checks["risk_exit_count"]:
            checks["warnings"].append("存在风险退出信号，必须结合完整资料人工复核")
    else:
        if len(actions) > loaded.profile.screening.max_new_funds:
            checks["hard_violations"].append("too many proposed new funds")
        checks["proposed_new_funds"] = len(actions)

    return {
        "status": "risk_checked",
        "research_actions": research_actions,
        "proposed_actions": actions,
        "risk_checks": checks,
        "trace": [
            _event(
                "RiskReviewAgent",
                f"完成当前组合集中度与研究信号检查；违规 {len(checks['hard_violations'])} 项",
            )
        ],
    }


def review_agent(state: AgentState) -> dict[str, Any]:
    hard_violations = state.get("risk_checks", {}).get("hard_violations", [])
    actions = state.get("proposed_actions", [])
    approval_required = (
        state["task_type"] in {"daily_rebalance", "weekly_screen"}
        and bool(actions)
        and not hard_violations
    )
    payload = {
        "run_id": state["run_id"],
        "task_type": state["task_type"],
        "instruction": "仅确认研究信号和研究池变更，不构成投资建议，不执行交易。",
        "research_actions": state.get("research_actions", []),
        "proposed_actions": actions,
        "target_weights": state.get("target_weights", {}),
        "risk_checks": state.get("risk_checks", {}),
        "reports": {
            "daily": state.get("analysis_report_path"),
            "weekly": state.get("candidate_report_path"),
            "monthly": state.get("monthly_report_path"),
        },
        "allowed_decisions": ["approve", "reject"],
    }
    if hard_violations:
        status = "blocked_by_risk"
    elif approval_required:
        status = "awaiting_approval"
    else:
        status = "reviewed"
    update = {
        "status": status,
        "approval_required": approval_required,
        "approval_payload": payload,
        "trace": [
            _event(
                "ReviewAgent",
                "发现硬风险违规，阻断审批"
                if hard_violations
                else "形成可审计研究信号并送人工确认"
                if approval_required
                else "复核完成，无需人工审批",
            )
        ],
    }
    merged = _merge_state_update(state, update)
    if state["task_type"] == "daily_rebalance":
        loaded = load_user_profile(state["profile_path"])
        ledger_path = loaded.resolve(loaded.profile.paper_ledger_file)
        if ledger_path is not None:
            try:
                recorded = record_run_recommendations(
                    ledger_path,
                    merged,
                    loaded.profile.validation.evaluation_horizons_days,
                )
                if recorded:
                    ledger_event = _event("PaperValidationLedger", f"记录 {recorded} 条前向研究信号")
                    update["trace"].append(ledger_event)
                    merged["trace"].append(ledger_event)
            except Exception as exc:
                ledger_event = _event(
                    "PaperTradingLedger",
                    f"模拟盘记录失败，不影响研究审批：{type(exc).__name__}: {exc}",
                )
                update["trace"].append(ledger_event)
                merged["trace"].append(ledger_event)
    _write_state_file(merged)
    return update


def human_approval_agent(state: AgentState) -> dict[str, Any]:
    try:
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError("Install agent dependencies with: pip install -e .[agent]") from exc
    response = interrupt(state["approval_payload"])
    decision = str(response.get("decision", "reject")).lower() if isinstance(response, dict) else "reject"
    if decision not in {"approve", "reject"}:
        decision = "reject"
    approval = dict(response) if isinstance(response, dict) else {"decision": decision}
    approval["decision"] = decision
    if decision == "approve" and state["task_type"] == "weekly_screen":
        _apply_approved_candidates(state)
    return {
        "status": "approved" if decision == "approve" else "rejected",
        "approval": approval,
        "trace": [
            _event(
                "HumanApprovalAgent",
                "用户确认研究结果；未执行真实交易" if decision == "approve" else "用户拒绝研究结果",
            )
        ],
    }


def finalize_agent(state: AgentState) -> dict[str, Any]:
    status = state.get("status", "completed")
    if status in {"reviewed", "monthly_research_completed"}:
        status = "completed"
    update = {
        "status": status,
        "trace": [_event("AuditAgent", f"运行结束，最终状态 {status}")],
    }
    _write_state_file(_merge_state_update(state, update))
    return update


@contextmanager
def _compiled_graph(loaded: LoadedProfile) -> Iterator[Any]:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError("Install agent dependencies with: pip install -e .[agent]") from exc
    state_dir = loaded.resolve(loaded.profile.state_dir)
    if state_dir is None:
        raise ValueError("state_dir is required")
    state_dir.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(state_dir / "checkpoints.sqlite")) as checkpointer:
        yield build_agent_graph(checkpointer)


def _effective_universe(loaded: LoadedProfile) -> tuple[FundSpec, ...]:
    base_path = loaded.resolve(loaded.profile.universe_file)
    if base_path is None:
        raise ValueError("universe_file is required")
    funds = {fund.code: fund for fund in load_fund_universe_csv(base_path)}
    approved_path = loaded.resolve(loaded.profile.investable_universe_file)
    if approved_path is not None and approved_path.exists() and approved_path.stat().st_size > 0:
        try:
            for fund in load_fund_universe_csv(approved_path):
                funds[fund.code] = fund
        except ValueError:
            pass
    return tuple(funds.values())


def _read_prices(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date")
    frame.index = pd.to_datetime(frame.index)
    frame.columns = [str(column).zfill(6) for column in frame.columns]
    return frame


def _serializable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.reset_index().to_dict(orient="records")
    return [
        {
            key: None if isinstance(value, float) and np.isnan(value) else value
            for key, value in record.items()
        }
        for record in records
    ]


def _read_context(state: AgentState) -> MarketContextSnapshot | None:
    if not state.get("context_path") or not state.get("context_meta_path"):
        return None
    signals = pd.read_csv(state["context_path"], dtype={"code": str}).set_index("code")
    signals.index = signals.index.astype(str).str.zfill(6)
    meta = json.loads(Path(state["context_meta_path"]).read_text(encoding="utf-8"))
    return MarketContextSnapshot(
        signals=signals,
        source=meta["source"],
        fetched_at=pd.Timestamp(meta["fetched_at"]),
        market_data_as_of=meta.get("market_data_as_of"),
        notes=tuple(meta.get("notes", [])),
    )


def _load_current_weights(loaded: LoadedProfile) -> pd.Series:
    portfolio_path = loaded.resolve(loaded.profile.portfolio_file)
    if portfolio_path is None or not portfolio_path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(portfolio_path, dtype={"code": str})
    if frame.empty:
        return pd.Series(dtype=float)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    weights = pd.to_numeric(
        frame.get("current_weight", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    values = pd.to_numeric(
        frame.get("current_value_yuan", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    if float(weights.fillna(0.0).sum()) > 0:
        current = pd.Series(weights.fillna(0.0).values, index=frame["code"])
    elif float(values.fillna(0.0).sum()) > 0:
        current = pd.Series(
            (values.fillna(0.0) / float(values.fillna(0.0).sum())).values,
            index=frame["code"],
        )
    else:
        return pd.Series(dtype=float)
    return current.groupby(level=0).sum()


def _portfolio_input_error(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    weights = pd.to_numeric(
        frame.get("current_weight", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    values = pd.to_numeric(
        frame.get("current_value_yuan", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    invalid = (weights <= 0.0) & (values <= 0.0)
    if not invalid.any():
        return None
    codes = (
        frame.loc[invalid, "code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    return (
        f"持仓中有 {int(invalid.sum())} 只基金缺少有效金额或权重：{', '.join(codes)}。"
        "请在“我的基金”中补全金额，或删除并未实际持有的行后再运行。"
    )


def _sample_screen_candidates(count: int) -> pd.DataFrame:
    rows = []
    categories = ["指数型-股票", "混合型-偏股", "债券型-中短债", "QDII-普通股票"]
    for index in range(count):
        rows.append(
            {
                "screen_rank": index + 1,
                "code": f"9{index + 1:05d}",
                "name": f"示例候选基金{index + 1}",
                "category": categories[index % len(categories)],
                "candidate_score": 0.95 - index * 0.02,
            }
        )
    return pd.DataFrame(rows)


def _update_candidate_watchlist(
    loaded: LoadedProfile,
    candidates: pd.DataFrame,
    observed_on: date,
) -> None:
    path = loaded.resolve(loaded.profile.candidate_watchlist_file)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["code", "name", "category", "score", "first_seen", "last_seen", "status"]
    current = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame(columns=columns)
    if not current.empty:
        current["code"] = current["code"].astype(str).str.zfill(6)
    score_column = "deep_score" if "deep_score" in candidates else "candidate_score"
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        existing = current["code"] == code if not current.empty else pd.Series(dtype=bool)
        values = {
            "code": code,
            "name": row["name"],
            "category": row["category"],
            "score": float(row[score_column]),
            "last_seen": observed_on.isoformat(),
            "status": "candidate",
        }
        if existing.any():
            for key, value in values.items():
                current.loc[existing, key] = value
        else:
            values["first_seen"] = observed_on.isoformat()
            current = pd.concat([current, pd.DataFrame([values])], ignore_index=True)
    current.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8")


def _apply_approved_candidates(state: AgentState) -> None:
    loaded = load_user_profile(state["profile_path"])
    path = loaded.resolve(loaded.profile.investable_universe_file)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "code",
        "name",
        "category",
        "is_equity_like",
        "instrument_type",
        "status",
        "notes",
    ]
    current = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame(columns=columns)
    if not current.empty:
        current["code"] = current["code"].astype(str).str.zfill(6)
    for action in state.get("proposed_actions", []):
        if action.get("operation") != "add_to_research_universe":
            continue
        code = str(action["code"]).zfill(6)
        category = str(action["category"])
        row = {
            "code": code,
            "name": action["name"],
            "category": category,
            "is_equity_like": str(not any(word in category for word in ["债券", "货币", "bond"])).lower(),
            "instrument_type": "open_fund",
            "status": "human_approved",
            "notes": f"Added to research universe from reviewed run {state['run_id']}.",
        }
        current = current[current["code"] != code]
        current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    current.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8")


def _ensure_runtime_files(loaded: LoadedProfile) -> None:
    ensure_user_runtime_files(loaded)


def _write_state_file(state: dict[str, Any]) -> None:
    path = Path(state["run_dir"]) / "run_state.json"
    payload = {key: value for key, value in state.items() if key != "__interrupt__"}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _merge_state_update(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    merged = {**state, **update}
    if "trace" in update:
        merged["trace"] = [*state.get("trace", []), *update["trace"]]
    return merged


def _scheduled_datetime(state: AgentState) -> datetime:
    return datetime.fromisoformat(state["scheduled_for"])


def _safe_lookback(prices: pd.DataFrame, requested: int) -> int | None:
    if len(prices) < 8:
        return None
    return min(requested, max(5, len(prices) - 3))


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _event(agent: str, summary: str) -> dict[str, Any]:
    return {"agent": agent, "summary": summary, "timestamp": datetime.now().astimezone().isoformat()}


def _new_run_id(task_type: str, scheduled_for: datetime) -> str:
    stamp = scheduled_for.strftime("%Y%m%d-%H%M%S")
    return f"{task_type}-{stamp}-{uuid4().hex[:6]}"


def _run_directory(loaded: LoadedProfile, run_id: str) -> Path:
    _validate_run_id(run_id)
    output_root = loaded.resolve(loaded.profile.output_root)
    if output_root is None:
        raise ValueError("output_root is required")
    run_dir = (output_root / run_id).resolve()
    try:
        run_dir.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError("run_id escapes output_root") from exc
    return run_dir


def _thread_config(run_id: str) -> dict[str, dict[str, str]]:
    _validate_run_id(run_id)
    return {"configurable": {"thread_id": run_id}}


def _validate_run_id(run_id: str) -> str:
    if run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "run_id must be 1-128 ASCII letters, digits, dots, underscores or hyphens, "
            "and may not contain path separators."
        )
    return run_id


def _normalize_graph_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    interrupts = normalized.pop("__interrupt__", ())
    if interrupts:
        normalized["interrupts"] = [getattr(item, "value", item) for item in interrupts]
    return normalized
