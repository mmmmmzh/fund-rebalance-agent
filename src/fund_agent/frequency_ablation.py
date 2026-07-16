from __future__ import annotations

import argparse
from pathlib import Path

from fund_agent.backtest import walk_forward_backtest
from fund_agent.config import DEFAULT_RISK_PROFILE_NAME, get_risk_profile
from fund_agent.optimizer import (
    equal_weight,
    fixed_allocation,
    max_sharpe,
    min_variance,
    momentum_allocation,
    risk_parity,
)
from fund_agent.runtime import load_dataset_with_fallback
from fund_agent.universe import load_fund_universe_csv


ALLOCATORS = [
    equal_weight,
    fixed_allocation,
    momentum_allocation,
    risk_parity,
    min_variance,
    max_sharpe,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rebalance frequencies.")
    parser.add_argument("--source", default="sample", choices=["sample", "plugin"])
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--risk-profile",
        default=DEFAULT_RISK_PROFILE_NAME.value,
        choices=["conservative", "balanced", "aggressive"],
    )
    parser.add_argument("--lookback-days", type=int, default=126)
    parser.add_argument("--frequencies", nargs="+", default=["D", "ME", "QE"])
    parser.add_argument("--output", default="reports/generated/frequency_ablation_report.md")
    parser.add_argument("--no-fallback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = get_risk_profile(args.risk_profile)
    fund_universe = load_fund_universe_csv(args.universe_file) if args.universe_file else None
    dataset, data_source_label = load_dataset_with_fallback(
        args.source,
        args.start,
        args.end,
        fund_universe=fund_universe,
        no_fallback=args.no_fallback,
    )

    rows: list[dict[str, float | str]] = []
    for freq in args.frequencies:
        for allocator in ALLOCATORS:
            result = walk_forward_backtest(
                dataset.prices,
                dataset.fund_universe,
                profile,
                allocator,
                lookback_days=args.lookback_days,
                rebalance_freq=freq,
            )
            summary = result.summary
            rows.append(
                {
                    "frequency": freq,
                    "strategy": result.strategy,
                    "cumulative_return": summary["cumulative_return"],
                    "annual_return": summary["annual_return"],
                    "annual_volatility": summary["annual_volatility"],
                    "max_drawdown": summary["max_drawdown"],
                    "sharpe": summary["sharpe"],
                    "average_turnover": summary["average_turnover"],
                    "rebalance_count": summary["rebalance_count"],
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_report(
            rows,
            data_source_label=data_source_label,
            risk_profile=args.risk_profile,
            lookback_days=args.lookback_days,
        ),
        encoding="utf-8",
    )
    print(f"Generated frequency ablation report: {output}")
    best = max(rows, key=lambda row: float(row["sharpe"]))
    print(
        "Best Sharpe: "
        f"{best['frequency']} / {best['strategy']} / sharpe={float(best['sharpe']):.2f}"
    )
    return 0


def _render_report(
    rows: list[dict[str, float | str]],
    data_source_label: str,
    risk_profile: str,
    lookback_days: int,
) -> str:
    lines = [
        "# Rebalance Frequency Ablation",
        "",
        "This report compares rebalance frequencies for research only. It is not investment advice.",
        "",
        "## Configuration",
        "",
        f"- Data source: `{data_source_label}`",
        f"- Risk profile: `{risk_profile}`",
        f"- Lookback window: `{lookback_days}` trading days",
        "",
        "## Results",
        "",
        "| Frequency | Strategy | Cum.Return | Ann.Return | Ann.Vol | Max Drawdown | Sharpe | Avg.Turnover | Rebalances |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['frequency']} | {row['strategy']} | "
            f"{float(row['cumulative_return']):.2%} | "
            f"{float(row['annual_return']):.2%} | "
            f"{float(row['annual_volatility']):.2%} | "
            f"{float(row['max_drawdown']):.2%} | "
            f"{float(row['sharpe']):.2f} | "
            f"{float(row['average_turnover']):.2%} | "
            f"{float(row['rebalance_count']):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Daily rebalancing is a stress test for turnover and optimizer stability.",
            "- Monthly and quarterly frequencies are usually more realistic for many off-exchange funds.",
            "- Do not use sample-data numbers as investment evidence.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
