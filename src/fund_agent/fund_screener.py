from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"code", "name", "category", "candidate_score"}


def screen_fund_candidates(
    candidates: pd.DataFrame,
    top_n: int = 10,
    max_per_category: int = 3,
) -> pd.DataFrame:
    """Rank a caller-supplied local candidate table without network access."""

    missing = sorted(REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {missing}")
    frame = candidates.copy()
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["candidate_score"] = pd.to_numeric(frame["candidate_score"], errors="coerce")
    frame = frame.dropna(subset=["candidate_score"]).drop_duplicates("code", keep="first")
    frame = frame.sort_values(["candidate_score", "code"], ascending=[False, True])
    frame = frame.groupby("category", group_keys=False).head(max_per_category)
    frame = frame.head(top_n).reset_index(drop=True)
    frame.insert(0, "screen_rank", range(1, len(frame) + 1))
    return frame


def write_screening_report(path: str | Path, candidates: pd.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 离线候选研究样例",
        "",
        "> 本报告仅对本地或合成候选表排序，不代表全市场覆盖，不构成基金推荐。",
        "",
        "| 排名 | 代码 | 名称 | 类别 | 研究分数 |",
        "|---:|---|---|---|---:|",
    ]
    for index, row in candidates.iterrows():
        rank = int(row.get("screen_rank", index + 1))
        lines.append(
            f"| {rank} | {str(row['code']).zfill(6)} | {row['name']} | "
            f"{row['category']} | {float(row['candidate_score']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 方法限制",
            "",
            "- 分数仅用于演示候选排序和人工审批工作流。",
            "- 未检索实时行情、基金合同、销售状态、费率或第三方排名。",
            "- 新增研究标的不会触发申购、赎回或任何账户操作。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank a local candidate CSV offline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="user_data/candidate_screen.csv")
    parser.add_argument("--report", default="user_data/candidate_screen.md")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-per-category", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = pd.read_csv(args.input, dtype={"code": str})
    result = screen_fund_candidates(candidates, args.top_n, args.max_per_category)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    write_screening_report(args.report, result)
    print(f"Wrote {len(result)} offline candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
