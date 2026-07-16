from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import pandas as pd

from fund_agent.user_profile import is_user_data_path


MAX_UPLOAD_BYTES = 1_000_000
MAX_ROWS = 500
OUTPUT_COLUMNS = ["code", "name", "current_weight", "current_value_yuan", "notes"]

COLUMN_ALIASES = {
    "code": "code",
    "基金代码": "code",
    "name": "name",
    "基金名称": "name",
    "current_weight": "current_weight",
    "当前权重": "current_weight",
    "权重": "current_weight",
    "current_value_yuan": "current_value_yuan",
    "当前金额": "current_value_yuan",
    "当前金额（元）": "current_value_yuan",
    "持仓金额": "current_value_yuan",
    "notes": "notes",
    "备注": "notes",
}

IDENTITY_COLUMN_PATTERNS = (
    "account",
    "user_id",
    "username",
    "real_name",
    "address",
    "wechat",
    "qq",
    "phone",
    "mobile",
    "email",
    "id_card",
    "bank_card",
    "账号",
    "用户id",
    "用户名",
    "真实姓名",
    "姓名",
    "地址",
    "微信",
    "手机号",
    "电话",
    "邮箱",
    "身份证",
    "银行卡",
)


def parse_portfolio_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("CSV 文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV 文件不能超过 1 MB。")
    frame = _read_csv(content)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV 最多允许 {MAX_ROWS} 行。")
    original_columns = [str(column).strip() for column in frame.columns]
    forbidden = [column for column in original_columns if _looks_like_identity_column(column)]
    if forbidden:
        raise ValueError(f"CSV 包含不允许保存的账户或身份字段：{', '.join(forbidden)}")
    rename = {
        column: COLUMN_ALIASES[column]
        for column in original_columns
        if column in COLUMN_ALIASES
    }
    mapped_columns = [rename[column] for column in original_columns if column in rename]
    duplicates = sorted({column for column in mapped_columns if mapped_columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"CSV 中存在重复含义的列：{', '.join(duplicates)}")
    frame.columns = original_columns
    frame = frame.rename(columns=rename)
    if "code" not in frame:
        raise ValueError("CSV 必须包含 code 或 基金代码 列。")
    if "name" not in frame:
        raise ValueError("CSV 必须包含 name 或 基金名称 列。")
    if "current_weight" not in frame and "current_value_yuan" not in frame:
        raise ValueError("CSV 必须包含当前权重或当前金额。")
    frame = frame.reindex(columns=OUTPUT_COLUMNS)
    return validate_portfolio_frame(frame)


def validate_portfolio_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy().reindex(columns=OUTPUT_COLUMNS)
    clean["code"] = clean["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    valid_codes = clean["code"].str.fullmatch(r"\d{1,12}", na=False)
    if not valid_codes.all():
        raise ValueError("基金代码必须由 1-12 位数字组成。")
    clean["code"] = clean["code"].str.zfill(6)
    if clean["code"].duplicated().any():
        raise ValueError("CSV 中存在重复基金代码。")
    clean["name"] = clean["name"].fillna("").astype(str).str.strip().str.slice(0, 100)
    if (clean["name"] == "").any():
        raise ValueError("基金名称不能为空。")
    clean["notes"] = clean["notes"].fillna("").astype(str).str.slice(0, 300)
    sensitive_notes = clean["notes"].map(_contains_identity_value)
    if sensitive_notes.any():
        raise ValueError("备注中疑似包含手机号、邮箱、身份证或银行卡信息。")
    for column in ["current_weight", "current_value_yuan"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if (clean[column].dropna() < 0).any():
            raise ValueError("权重和金额不能为负数。")
    values = clean["current_value_yuan"]
    weights = clean["current_weight"]
    if values.notna().all() and float(values.sum()) > 0:
        clean["current_weight"] = values / float(values.sum())
    elif float(weights.fillna(0.0).sum()) > 1.000001:
        raise ValueError("当前权重合计不能超过 100%。")
    invalid = (clean["current_weight"].fillna(0.0) <= 0) & (
        clean["current_value_yuan"].fillna(0.0) <= 0
    )
    if invalid.any():
        codes = ", ".join(clean.loc[invalid, "code"].tolist())
        raise ValueError(f"以下基金缺少有效权重或金额：{codes}")
    return clean


def save_portfolio_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path).resolve()
    if not is_user_data_path(target):
        raise ValueError("持仓 CSV 只能写入允许的 user_data 目录。")
    clean = validate_portfolio_frame(frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(target, index=False, encoding="utf-8")
    return target


def _read_csv(content: bytes) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding, dtype=str)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("无法解析 CSV，请使用 UTF-8 或 GB18030 编码。" + "; ".join(errors))


def _looks_like_identity_column(column: str) -> bool:
    normalized = re.sub(r"[\s\-]+", "_", column.strip().lower())
    return any(pattern in normalized for pattern in IDENTITY_COLUMN_PATTERNS)


def _contains_identity_value(value: str) -> bool:
    patterns = (
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        r"(?<!\d)\d{17}[0-9Xx](?!\d)",
        r"(?<!\d)\d{16,19}(?!\d)",
    )
    return any(re.search(pattern, value) for pattern in patterns)
