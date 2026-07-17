from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fund_agent.paper_trading import load_paper_ledger
from fund_agent.portfolio_csv import parse_portfolio_csv, save_portfolio_csv, validate_portfolio_frame
from fund_agent.user_profile import (
    PROJECT_ROOT,
    clone_user_workspace,
    ensure_user_runtime_files,
    initialize_user_workspace,
    is_user_data_path,
    load_user_profile,
)


TASK_LABELS = {
    "daily_rebalance": "每日研究",
    "weekly_screen": "周五候选",
    "monthly_research": "月度主题",
}
STATUS_LABELS = {
    "awaiting_approval": "待人工确认",
    "approved": "已确认",
    "rejected": "已拒绝",
    "completed": "已完成",
    "blocked_by_risk": "风险阻断",
    "running": "运行中",
}


def main() -> None:
    st.set_page_config(page_title="Fund Rebalance Agent", layout="wide")
    st.markdown(
        """
        <style>
        @media (max-width: 700px) {
          [data-baseweb="tab-list"] {
            flex-wrap: wrap;
            row-gap: 0.25rem;
          }
          [data-baseweb="tab"] {
            flex: 1 1 auto;
            min-width: max-content;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Fund Rebalance Agent")
    st.caption("离线基金研究工作台 · 合成数据 · 人工确认 · 不连接交易账户")

    profile_path = _profile_selector()
    if profile_path is None:
        _profile_initializer()
        return
    try:
        loaded = load_user_profile(profile_path)
        ensure_user_runtime_files(loaded)
    except Exception as exc:
        st.error(f"配置加载失败：{type(exc).__name__}: {exc}")
        return

    overview = _profile_overview(loaded)
    cols = st.columns(4)
    cols[0].metric("研究池", overview["universe_count"])
    cols[1].metric("本地持仓", overview["portfolio_count"])
    cols[2].metric("数据模式", "离线合成" if loaded.profile.data_source == "sample" else "插件")
    cols[3].metric("人工确认", "必需")

    task_tab, funds_tab, validation_tab, history_tab, settings_tab = st.tabs(
        ["任务", "本地数据", "前向验证", "运行历史", "边界"]
    )
    with task_tab:
        _task_panel(profile_path, loaded)
    with funds_tab:
        _data_panel(loaded)
    with validation_tab:
        _validation_panel(loaded)
    with history_tab:
        _history_panel(loaded)
    with settings_tab:
        _settings_panel(loaded)


def _profile_selector() -> Path | None:
    profiles = _discover_profiles()
    with st.sidebar:
        st.header("本地配置")
        if not profiles:
            st.info("尚无可用配置。")
            return None
        labels = {str(path): _profile_label(path) for path in profiles}
        selected = st.selectbox("配置", list(labels), format_func=labels.get)
        st.caption("仅列出 config/ 与 user_data/ 中通过校验的配置。")
    return Path(selected)


def _discover_profiles() -> list[Path]:
    candidates = [PROJECT_ROOT / "config" / "demo_profile.json"]
    candidates.extend((PROJECT_ROOT / "user_data").glob("*/profile.json"))
    result = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            load_user_profile(path)
        except Exception:
            continue
        result.append(path.resolve())
    return result


def _profile_label(path: Path) -> str:
    try:
        profile = load_user_profile(path).profile
        suffix = "公开演示" if "config" in path.parts else "本地"
        return f"{profile.profile_name} · {suffix}"
    except Exception:
        return path.name


def _profile_initializer() -> None:
    with st.form("new-profile"):
        st.subheader("创建本地研究空间")
        name = st.text_input("名称", "my-research")
        submitted = st.form_submit_button("创建", type="primary")
    if submitted:
        slug = "".join(character for character in name.lower() if character.isalnum() or character in "_-")
        if not slug:
            st.error("名称至少包含一个字母或数字。")
            return
        try:
            initialize_user_workspace(PROJECT_ROOT / "user_data" / slug, name)
        except Exception as exc:
            st.error(f"创建失败：{type(exc).__name__}: {exc}")
            return
        st.rerun()


def _task_panel(profile_path: Path, loaded) -> None:
    st.subheader("运行研究任务")
    columns = st.columns(3)
    selected_task = None
    if columns[0].button("每日研究", type="primary", width="stretch"):
        selected_task = "daily_rebalance"
    if columns[1].button("周五候选", width="stretch"):
        selected_task = "weekly_screen"
    if columns[2].button("月度主题", width="stretch"):
        selected_task = "monthly_research"
    if selected_task:
        try:
            from fund_agent.agent_workflow import run_agent_workflow

            with st.spinner("Agent 正在运行..."):
                result = run_agent_workflow(profile_path, selected_task)
            st.session_state["latest_agent_result"] = result
        except Exception as exc:
            st.error(f"任务失败：{type(exc).__name__}: {exc}")

    result = st.session_state.get("latest_agent_result")
    if result:
        _show_result(profile_path, result)
    else:
        latest = _load_historical_run_states(loaded, limit=1)
        if latest:
            _show_result(profile_path, latest[0], historical=True)


def _show_result(profile_path: Path, result: dict[str, Any], historical: bool = False) -> None:
    status = str(result.get("status", "unknown"))
    st.markdown(f"**{TASK_LABELS.get(result.get('task_type'), result.get('task_type', '任务'))}**")
    st.write(STATUS_LABELS.get(status, status))
    if result.get("fund_analyses"):
        frame = pd.DataFrame(result["fund_analyses"])
        columns = [
            column
            for column in [
                "code",
                "name",
                "signal",
                "decision_score",
                "decision_confidence",
                "return_20d",
                "max_drawdown_60d",
            ]
            if column in frame
        ]
        st.dataframe(frame[columns], width="stretch", hide_index=True)
    signals = result.get("proposed_actions", [])
    if signals:
        st.markdown("**待复核研究项**")
        st.dataframe(_signal_display_frame(signals), width="stretch", hide_index=True)
    checks = result.get("risk_checks", {})
    for warning in checks.get("warnings", []):
        st.warning(warning)
    for violation in checks.get("hard_violations", []):
        st.error(violation)
    report_paths = [
        result.get("analysis_report_path"),
        result.get("candidate_report_path"),
        result.get("monthly_report_path"),
    ]
    for report in report_paths:
        if report and Path(report).exists():
            with st.expander(Path(report).name):
                st.markdown(Path(report).read_text(encoding="utf-8"))
    if status == "awaiting_approval" and not historical:
        left, right = st.columns(2)
        if left.button("确认研究结果", type="primary", width="stretch"):
            _resume(profile_path, result["run_id"], "approve")
        if right.button("拒绝研究结果", width="stretch"):
            _resume(profile_path, result["run_id"], "reject")


def _resume(profile_path: Path, run_id: str, decision: str) -> None:
    try:
        from fund_agent.agent_workflow import resume_agent_workflow

        result = resume_agent_workflow(profile_path, run_id, decision)
        st.session_state["latest_agent_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"确认失败：{type(exc).__name__}: {exc}")


def _data_panel(loaded) -> None:
    portfolio_path = loaded.resolve(loaded.profile.portfolio_file)
    if not _is_private_user_path(portfolio_path):
        st.info("公开演示配置为只读。先创建本地副本再导入自己的研究持仓。")
        if st.button("创建本地副本", type="primary"):
            target = PROJECT_ROOT / "user_data" / "my-research"
            try:
                clone_user_workspace(loaded, target)
            except FileExistsError:
                pass
            st.rerun()
        _show_csv(portfolio_path)
        return

    st.subheader("导入持仓 CSV")
    upload = st.file_uploader("选择 CSV", type=["csv"], accept_multiple_files=False)
    if upload and st.button("校验并导入", type="primary"):
        try:
            frame = parse_portfolio_csv(upload.getvalue())
            save_portfolio_csv(frame, portfolio_path)
        except Exception as exc:
            st.error(f"导入失败：{exc}")
        else:
            st.success("已保存标准化字段；原文件名和其他列未保存。")
            st.rerun()

    frame = _read_csv(portfolio_path).reindex(
        columns=["code", "name", "current_weight", "current_value_yuan", "notes"]
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "current_weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.0),
            "current_value_yuan": st.column_config.NumberColumn(min_value=0.0),
        },
    )
    if st.button("保存表格"):
        error = _validate_and_save_portfolio(edited, portfolio_path)
        st.error(error) if error else st.success("已保存。")

    st.markdown("**CSV 字段**")
    st.code("code,name,current_weight,current_value_yuan,notes", language="text")
    st.caption("不要包含账号、姓名、手机号、邮箱、身份证或银行卡等身份字段。")


def _validation_panel(loaded) -> None:
    path = loaded.resolve(loaded.profile.paper_ledger_file)
    ledger = load_paper_ledger(path) if path is not None else pd.DataFrame()
    if ledger.empty:
        st.info("运行每日研究后，这里会记录信号的前向表现。")
        return
    cols = st.columns(3)
    cols[0].metric("信号记录", len(ledger))
    for index, horizon in enumerate((5, 20), start=1):
        column = f"signal_return_{horizon}d"
        available = pd.to_numeric(ledger[column], errors="coerce").dropna()
        hit_rate = float((available > 0).mean()) if not available.empty else 0.0
        cols[index].metric(f"{horizon} 日方向命中", f"{hit_rate:.0%}" if not available.empty else "待观察")
    st.dataframe(ledger, width="stretch", hide_index=True)


def _history_panel(loaded) -> None:
    states = _load_historical_run_states(loaded)
    if not states:
        st.info("暂无运行历史。")
        return
    overview = pd.DataFrame([_run_overview_row(state) for state in states])
    st.dataframe(overview, width="stretch", hide_index=True)
    options = {state["run_id"]: state for state in states if state.get("run_id")}
    selected = st.selectbox("查看任务", list(options))
    if selected:
        _show_result(loaded.path, options[selected], historical=True)


def _settings_panel(loaded) -> None:
    profile = loaded.profile
    st.subheader("研究设置")
    st.json(
        {
            "risk_profile": profile.risk_profile,
            "lookback_days": profile.lookback_days,
            "rebalance_freq": profile.rebalance_freq,
            "data_source": profile.data_source,
            "market_context_source": profile.market_context_source,
            "human_approval": "required",
        }
    )
    st.warning(
        "本项目是本地研究与软件演示，不提供个性化投资建议，不承诺收益，"
        "不连接交易账户，也不会自动申购或赎回。"
    )


def _profile_overview(loaded) -> dict[str, int]:
    universe = _read_csv(loaded.resolve(loaded.profile.universe_file))
    portfolio = _read_csv(loaded.resolve(loaded.profile.portfolio_file))
    return {"universe_count": len(universe), "portfolio_count": len(portfolio)}


def _load_historical_run_states(loaded, limit: int = 50) -> list[dict[str, Any]]:
    output_root = loaded.resolve(loaded.profile.output_root)
    if output_root is None or not output_root.exists():
        return []
    states = []
    for path in output_root.glob("*/run_state.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_mtime"] = path.stat().st_mtime
            states.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    states.sort(key=lambda item: item.get("_mtime", 0), reverse=True)
    return states[:limit]


def _run_overview_row(state: dict[str, Any]) -> dict[str, Any]:
    trace = state.get("trace", [])
    duration = None
    if len(trace) >= 2:
        try:
            start = datetime.fromisoformat(trace[0]["timestamp"])
            end = datetime.fromisoformat(trace[-1]["timestamp"])
            duration = round((end - start).total_seconds(), 1)
        except (KeyError, TypeError, ValueError):
            pass
    checks = state.get("risk_checks", {})
    return {
        "任务 ID": state.get("run_id"),
        "类型": TASK_LABELS.get(state.get("task_type"), state.get("task_type")),
        "计划时间": state.get("scheduled_for"),
        "状态": STATUS_LABELS.get(state.get("status"), state.get("status")),
        "研究项": len(state.get("proposed_actions", [])),
        "硬风险": len(checks.get("hard_violations", [])),
        "提示": len(checks.get("warnings", [])),
        "耗时（秒）": duration,
    }


def _signal_display_frame(signals: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(signals)
    if frame.empty:
        return frame
    labels = {"strong": "偏强", "weak": "偏弱", "risk_exit": "风险退出"}
    if "signal" in frame:
        frame["signal"] = frame["signal"].map(labels).fillna(frame["signal"])
    return frame


def _validate_and_save_portfolio(frame: pd.DataFrame, path: Path | None) -> str | None:
    if path is None:
        return "未配置持仓文件。"
    try:
        save_portfolio_csv(validate_portfolio_frame(frame), path)
    except Exception as exc:
        return str(exc)
    return None


def _is_private_user_path(path: Path | None) -> bool:
    return is_user_data_path(path)


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"code": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _show_csv(path: Path | None) -> None:
    frame = _read_csv(path)
    if frame.empty:
        st.info("暂无数据。")
    else:
        st.dataframe(frame, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
