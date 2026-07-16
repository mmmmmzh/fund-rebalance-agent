from __future__ import annotations

import pytest

from fund_agent.adapters import reset_adapters


@pytest.fixture(autouse=True)
def isolated_user_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FUND_AGENT_USER_DATA_ROOT", str(tmp_path))
    reset_adapters()
    yield
    reset_adapters()
