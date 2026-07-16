from __future__ import annotations

import pandas as pd

from fund_agent.skills import (
    aggregate_skill_signal,
    load_skill_definitions,
    run_fund_skills,
    save_user_skill,
)


def test_builtin_skills_produce_per_fund_signals() -> None:
    facts = pd.DataFrame(
        [
            {
                "code": "000001",
                "return_20d": 0.08,
                "return_60d": 0.12,
                "annual_volatility_60d": 0.20,
                "max_drawdown_60d": -0.05,
                "context_score": 0.3,
                "context_confidence": 0.8,
            }
        ]
    ).set_index("code")
    definitions = load_skill_definitions()

    outputs, errors = run_fund_skills(
        facts,
        definitions,
        ["momentum_confirmation", "drawdown_guard", "market_context_consensus"],
    )
    score, confidence = aggregate_skill_signal(outputs["000001"])

    assert not errors
    assert len(outputs["000001"]) == 3
    assert -1.0 <= score <= 1.0
    assert 0.0 < confidence <= 1.0


def test_user_skill_is_json_only_and_can_override_by_id(tmp_path) -> None:
    output = save_user_skill(
        tmp_path,
        {
            "skill_id": "policy_cycle_review",
            "name": "政策周期复核",
            "description": "复核市场环境",
            "settings": {"mode": "cycle-review"},
            "weight": 1.2,
        },
    )

    definitions = load_skill_definitions(tmp_path)
    saved = next(item for item in definitions if item.skill_id == "policy_cycle_review")

    assert output.suffix == ".json"
    assert saved.kind == "plugin"
    assert saved.handler is None
    assert saved.weight == 1.2
