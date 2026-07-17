from __future__ import annotations

import json

import pytest

from fund_agent.user_profile import (
    clone_user_workspace,
    ensure_user_runtime_files,
    initialize_user_workspace,
    load_user_profile,
    validate_user_profile,
)


def test_initialize_and_validate_local_user_workspace(tmp_path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "alice", "alice")

    loaded = load_user_profile(profile_path)
    warnings = validate_user_profile(loaded)

    assert loaded.profile.profile_name == "alice"
    assert loaded.profile.human_approval_required
    assert loaded.profile.screening.portfolio_gap_mode == "diversification_only"
    assert loaded.profile.investment_policy.max_acceptable_drawdown == 0.20
    assert loaded.profile.validation.evaluation_horizons_days == [5, 20, 60]
    assert not warnings
    assert loaded.resolve(loaded.profile.universe_file).exists()
    assert loaded.resolve(loaded.profile.portfolio_file).exists()
    assert loaded.resolve(loaded.profile.paper_ledger_file).exists()


def test_profile_rejects_invalid_schedule_time(tmp_path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "bob", "bob")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["schedule"]["daily_time"] = "25:90"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_user_profile(profile_path)
    except ValueError as exc:
        assert "Time must use HH:MM" in str(exc)
    else:
        raise AssertionError("Invalid schedule time was accepted")


def test_profile_rejects_disabled_human_approval_with_migration_error(tmp_path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "approval", "approval")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["human_approval_required"] = False
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="公开版强制人工确认"):
        load_user_profile(profile_path)


def test_ensure_runtime_files_creates_private_csvs(tmp_path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "carol", "carol")
    loaded = load_user_profile(profile_path)
    portfolio_path = loaded.resolve(loaded.profile.portfolio_file)
    portfolio_path.unlink()

    created = ensure_user_runtime_files(loaded)

    assert set(created) == {portfolio_path}
    assert portfolio_path.read_text(encoding="utf-8").startswith("code,name,current_weight")


def test_clone_workspace_preserves_settings_and_makes_files_local(tmp_path) -> None:
    source_path = initialize_user_workspace(tmp_path / "source", "source")
    source = load_user_profile(source_path)

    cloned_path = clone_user_workspace(source, tmp_path / "clone")
    cloned = load_user_profile(cloned_path)

    assert cloned.profile.profile_name == "source"
    assert cloned.profile.universe_file == "universe.csv"
    assert cloned.profile.output_root == "runs"
    assert cloned.resolve(cloned.profile.universe_file).exists()
    assert not validate_user_profile(cloned)


def test_profile_rejects_writable_path_outside_user_data(tmp_path) -> None:
    profile_path = initialize_user_workspace(tmp_path / "escape-test", "escape-test")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["output_root"] = "../../outside"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="output_root"):
        load_user_profile(profile_path)


def test_profile_rejects_loading_from_arbitrary_directory(tmp_path, monkeypatch) -> None:
    profile_path = initialize_user_workspace(tmp_path / "allowed", "allowed")
    monkeypatch.delenv("FUND_AGENT_USER_DATA_ROOT")

    with pytest.raises(ValueError, match="Profiles may only be loaded"):
        load_user_profile(profile_path)
