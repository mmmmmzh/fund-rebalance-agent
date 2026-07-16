from __future__ import annotations

import argparse
from pathlib import Path

from fund_agent.config import DEFAULT_LOOKBACK_DAYS, DEFAULT_REBALANCE_FREQ, DEFAULT_RISK_PROFILE_NAME, get_risk_profile
from fund_agent.market_context import load_market_context
from fund_agent.pipeline import run_analysis
from fund_agent.reporting import write_markdown_report
from fund_agent.runtime import load_dataset_with_fallback
from fund_agent.universe import load_fund_universe_csv
from fund_agent.visualization import generate_report_charts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FundRebalance-Agent MVP.")
    parser.add_argument("--source", default="sample", choices=["sample", "plugin"])
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--risk-profile",
        default=DEFAULT_RISK_PROFILE_NAME.value,
        choices=["conservative", "balanced", "aggressive"],
    )
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--rebalance-freq", default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument(
        "--market-context",
        default="auto",
        choices=["auto", "none", "sample", "plugin"],
        help="Current-market context used only for the latest target weights.",
    )
    parser.add_argument("--output", default="reports/generated/daily_aggressive_rebalance_report.md")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back to sample data when an external data source is unavailable.",
    )
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

    context_source = dataset.source if args.market_context == "auto" else args.market_context
    market_context = None
    if context_source != "none":
        try:
            market_context = load_market_context(
                context_source,
                dataset.fund_universe,
                dataset.prices,
            )
        except Exception as exc:
            if args.no_fallback:
                raise
            print(
                f"Warning: market context {context_source!r} failed with "
                f"{type(exc).__name__}: {exc}. Continuing without current-market context."
            )

    analysis = run_analysis(
        dataset,
        profile,
        args.lookback_days,
        args.rebalance_freq,
        market_context=market_context,
    )
    output_path_arg = Path(args.output)
    chart_paths = generate_report_charts(
        output_path_arg.parent / "assets" / output_path_arg.stem,
        analysis.backtests,
        analysis.target_weights,
    )
    output_path = write_markdown_report(
        output_path_arg,
        dataset.prices,
        dataset.fund_universe,
        profile,
        analysis.briefs,
        analysis.backtests,
        analysis.target_weights,
        data_source_label,
        args.rebalance_freq,
        args.lookback_days,
        chart_paths,
        forecast=analysis.forecast,
        target_strategy=analysis.target_strategy,
    )
    print(f"Generated report: {output_path}")
    print("Backtest summary:")
    for result in analysis.backtests:
        summary = result.summary
        print(
            f"- {result.strategy}: cumulative={summary['cumulative_return']:.2%}, "
            f"sharpe={summary['sharpe']:.2f}, max_drawdown={summary['max_drawdown']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
