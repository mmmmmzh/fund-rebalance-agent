from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from fund_agent.adapters import get_adapters, require_adapter
from fund_agent.user_profile import is_user_data_path


BUILTIN_SKILL_PAYLOADS = (
    {
        "skill_id": "momentum_confirmation",
        "name": "多周期趋势确认",
        "description": "比较近 20 日与 60 日收益，识别趋势方向是否一致。",
        "kind": "builtin",
        "handler": "momentum_confirmation",
        "weight": 1.0,
    },
    {
        "skill_id": "drawdown_guard",
        "name": "回撤风险哨兵",
        "description": "根据近 60 日最大回撤与波动识别风险压力。",
        "kind": "builtin",
        "handler": "drawdown_guard",
        "weight": 1.0,
    },
    {
        "skill_id": "market_context_consensus",
        "name": "市场信号一致性",
        "description": "汇总离线市场上下文或显式插件信号及其置信度。",
        "kind": "builtin",
        "handler": "market_context_consensus",
        "weight": 1.0,
    },
)


class SkillDefinition(BaseModel):
    skill_id: str
    name: str
    description: str
    kind: Literal["builtin", "plugin"] = "builtin"
    handler: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0.0, le=2.0)

    @field_validator("skill_id")
    @classmethod
    def validate_skill_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized or not all(char.isalnum() or char in "_-" for char in normalized):
            raise ValueError("skill_id may only contain letters, numbers, underscore, and hyphen")
        return normalized

    @model_validator(mode="after")
    def validate_implementation(self) -> "SkillDefinition":
        if self.kind == "builtin" and not self.handler:
            raise ValueError("builtin skills require handler")
        return self


def load_skill_definitions(user_skills_dir: str | Path | None = None) -> list[SkillDefinition]:
    definitions: dict[str, SkillDefinition] = {}
    for payload in BUILTIN_SKILL_PAYLOADS:
        definition = SkillDefinition.model_validate(payload)
        definitions[definition.skill_id] = definition
    if user_skills_dir is not None:
        directory = Path(user_skills_dir)
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                definition = SkillDefinition.model_validate_json(path.read_text(encoding="utf-8"))
                if definition.kind != "plugin":
                    raise ValueError(f"User skill must use kind=plugin: {path}")
                definitions[definition.skill_id] = definition
    return list(definitions.values())


def save_user_skill(user_skills_dir: str | Path, payload: dict[str, Any]) -> Path:
    definition = SkillDefinition.model_validate({**payload, "kind": "plugin", "handler": None})
    directory = Path(user_skills_dir)
    if not is_user_data_path(directory):
        raise ValueError("Plugin skill definitions must stay inside user_data/.")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{definition.skill_id}.json"
    output.write_text(
        json.dumps(definition.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def run_fund_skills(
    facts: pd.DataFrame,
    definitions: list[SkillDefinition],
    enabled_skill_ids: list[str],
    **_: object,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    enabled = {skill_id.lower() for skill_id in enabled_skill_ids}
    outputs: dict[str, list[dict[str, Any]]] = {str(code): [] for code in facts.index}
    errors: list[str] = []
    for definition in definitions:
        if definition.skill_id not in enabled:
            continue
        try:
            signals = (
                _run_builtin_skill(definition, facts)
                if definition.kind == "builtin"
                else require_adapter("skill", get_adapters().skill)(
                    definition.skill_id, facts, definition.settings
                )
            )
        except Exception as exc:
            errors.append(f"{definition.skill_id}: {type(exc).__name__}: {exc}")
            continue
        for code, signal in signals.items():
            if code not in outputs:
                continue
            outputs[code].append(
                {
                    "skill_id": definition.skill_id,
                    "skill_name": definition.name,
                    "score": float(np.clip(signal["score"], -1.0, 1.0)),
                    "confidence": float(np.clip(signal["confidence"], 0.0, 1.0)),
                    "weight": definition.weight,
                    "summary": str(signal["summary"])[:300],
                    "risk": str(signal.get("risk", ""))[:300],
                }
            )
    return outputs, errors


def aggregate_skill_signal(signals: list[dict[str, Any]]) -> tuple[float, float]:
    weighted_sum = 0.0
    effective_weight = 0.0
    for signal in signals:
        weight = float(signal["weight"]) * float(signal["confidence"])
        weighted_sum += float(signal["score"]) * weight
        effective_weight += weight
    if effective_weight <= 0:
        return 0.0, 0.0
    return float(np.clip(weighted_sum / effective_weight, -1.0, 1.0)), float(
        np.clip(effective_weight / max(1, len(signals)), 0.0, 1.0)
    )


def _run_builtin_skill(
    definition: SkillDefinition,
    facts: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    handlers = {
        "momentum_confirmation": _momentum_confirmation,
        "drawdown_guard": _drawdown_guard,
        "market_context_consensus": _market_context_consensus,
    }
    handler = handlers.get(str(definition.handler))
    if handler is None:
        raise ValueError(f"Unknown builtin skill handler: {definition.handler}")
    return {str(code): handler(row) for code, row in facts.iterrows()}


def _momentum_confirmation(row: pd.Series) -> dict[str, Any]:
    return_20d = _number(row.get("return_20d"))
    return_60d = _number(row.get("return_60d"))
    available = [value for value in (return_20d, return_60d) if value is not None]
    if not available:
        return {"score": 0.0, "confidence": 0.0, "summary": "缺少多周期收益数据。"}
    score = 0.6 * np.tanh((return_20d or 0.0) / 0.08) + 0.4 * np.tanh(
        (return_60d or 0.0) / 0.15
    )
    direction = "偏强" if score > 0.2 else "偏弱" if score < -0.2 else "震荡"
    return {
        "score": score,
        "confidence": len(available) / 2,
        "summary": f"20 日 {return_20d or 0:+.1%}，60 日 {return_60d or 0:+.1%}，趋势{direction}。",
    }


def _drawdown_guard(row: pd.Series) -> dict[str, Any]:
    drawdown = _number(row.get("max_drawdown_60d"))
    volatility = _number(row.get("annual_volatility_60d"))
    if drawdown is None:
        return {"score": 0.0, "confidence": 0.0, "summary": "缺少回撤数据。"}
    pressure = min(abs(drawdown) / 0.25, 1.0)
    return {
        "score": -pressure,
        "confidence": 0.9 if volatility is not None else 0.7,
        "summary": f"近 60 日最大回撤 {drawdown:.1%}。",
        "risk": f"年化波动 {volatility:.1%}" if volatility is not None else "",
    }


def _market_context_consensus(row: pd.Series) -> dict[str, Any]:
    score = _number(row.get("context_score"))
    confidence = _number(row.get("context_confidence"))
    if score is None or confidence is None:
        return {"score": 0.0, "confidence": 0.0, "summary": "市场上下文不可用。"}
    direction = "正向" if score > 0.15 else "负向" if score < -0.15 else "中性"
    return {
        "score": score,
        "confidence": confidence,
        "summary": f"离线市场上下文{direction}，置信度 {confidence:.0%}。",
    }


def _number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
