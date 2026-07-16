from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from fund_agent.backtest import BacktestResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_report_charts(
    output_dir: str | Path,
    backtests: list[BacktestResult],
    target_weights: pd.Series,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart_paths = {
        "equity_curve": output / "equity_curve.png",
        "drawdown_curve": output / "drawdown_curve.png",
        "target_weights": output / "target_weights.png",
    }
    _plot_equity_curves(chart_paths["equity_curve"], backtests)
    _plot_drawdowns(chart_paths["drawdown_curve"], backtests)
    _plot_target_weights(chart_paths["target_weights"], target_weights)
    return chart_paths


def _plot_equity_curves(path: Path, backtests: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in backtests:
        ax.plot(result.equity_curve.index, result.equity_curve.values, label=result.strategy)
    ax.set_title("Walk-forward Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_drawdowns(path: Path, backtests: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in backtests:
        curve = result.equity_curve
        drawdown = curve / curve.cummax() - 1.0
        ax.plot(drawdown.index, drawdown.values, label=result.strategy)
    ax.set_title("Walk-forward Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_target_weights(path: Path, target_weights: pd.Series) -> None:
    weights = target_weights.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(weights.index, weights.values)
    ax.set_title("Target Weights")
    ax.set_xlabel("Weight")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

