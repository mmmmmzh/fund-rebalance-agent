from __future__ import annotations

import pandas as pd
import pytest

from fund_agent.portfolio_csv import parse_portfolio_csv, save_portfolio_csv


def test_import_keeps_only_allowlisted_fields_and_normalizes_weights(tmp_path) -> None:
    content = (
        "基金代码,基金名称,当前金额（元）,备注,自定义列\n"
        "51,示例A,3000,研究,discard\n"
        "52,示例B,1000,,discard\n"
    ).encode("utf-8")

    frame = parse_portfolio_csv(content)
    path = save_portfolio_csv(frame, tmp_path / "portfolio.csv")
    saved = pd.read_csv(path, dtype={"code": str})

    assert saved.columns.tolist() == [
        "code",
        "name",
        "current_weight",
        "current_value_yuan",
        "notes",
    ]
    assert saved["current_weight"].tolist() == [0.75, 0.25]
    assert "自定义列" not in saved


@pytest.mark.parametrize("column", ["账号", "phone", "email", "身份证号", "bank_card"])
def test_import_rejects_identity_columns(column: str) -> None:
    content = f"code,name,current_weight,{column}\n1,示例,1.0,secret\n".encode()

    with pytest.raises(ValueError, match="身份字段"):
        parse_portfolio_csv(content)


def test_import_rejects_oversized_upload() -> None:
    with pytest.raises(ValueError, match="1 MB"):
        parse_portfolio_csv(b"x" * 1_000_001)


@pytest.mark.parametrize(
    "note",
    ["联系 13800138000", "mail test@example.com", "身份证 11010519491231002X", "卡号 6222021234567890"],
)
def test_import_rejects_identity_values_in_notes(note: str) -> None:
    content = f"code,name,current_weight,notes\n1,示例,1.0,{note}\n".encode()

    with pytest.raises(ValueError, match="备注中疑似包含"):
        parse_portfolio_csv(content)
