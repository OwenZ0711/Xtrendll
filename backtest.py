"""
Backtest utilities: metrics, equity curves, comparison plots.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .train import annualised_sharpe, max_drawdown


def annualised_return(daily: pd.Series) -> float:
    x = daily.dropna().values.astype(np.float64)
    if len(x) == 0:
        return np.nan
    return float(np.prod(1 + x) ** (252 / len(x)) - 1)


def annualised_vol(daily: pd.Series) -> float:
    x = daily.dropna().values.astype(np.float64)
    return float(np.std(x, ddof=0) * np.sqrt(252)) if len(x) else np.nan


def calmar(daily: pd.Series, eps=1e-9) -> float:
    return annualised_return(daily) / max(abs(max_drawdown(daily)), eps)


def summary(daily: pd.Series) -> Dict[str, float]:
    return {
        "ann_return":  annualised_return(daily),
        "ann_vol":     annualised_vol(daily),
        "sharpe":      annualised_sharpe(daily),
        "max_dd":      max_drawdown(daily),
        "calmar":      calmar(daily),
        "hit_rate":    float((daily.dropna() > 0).mean()) if len(daily) else np.nan,
        "days":        len(daily.dropna()),
    }


def turnover(pred_df: pd.DataFrame) -> pd.Series:
    """Average daily absolute position change across tickers."""
    df = pred_df.sort_values(["ticker", "date"]).copy()
    df["prev_pos"] = df.groupby("ticker")["position"].shift(1).fillna(0)
    df["to"] = (df["position"] - df["prev_pos"]).abs()
    return df.groupby("date")["to"].mean().sort_index()

def build_benchmarks(pred_df: pd.DataFrame) -> list[Dict]:
    """
    Build equal-weight and SPY-only benchmark results
    using the same dates and returns from the model's pred_df.
    """
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    tickers = df["ticker"].unique()
    n = len(tickers)

    # ── Benchmark 1: Equal-weight (1/N) ──
    ew = df[["date", "ticker", "target_return"]].copy()
    ew["position"] = 1.0 / n

    # ── Benchmark 2: SPY buy-and-hold ──
    spy = df[df["ticker"] == "SPY"][["date", "ticker", "target_return"]].copy()
    spy["position"] = 1.0

    benchmarks = []
    from .config import TRAIN
    cost = TRAIN["cost_bps"]

    benchmarks.append(run_backtest(ew,  cost_bps=cost, label=f"Equal-Weight 1/{n}"))
    benchmarks.append(run_backtest(spy, cost_bps=cost, label="SPY Buy-&-Hold"))

    return benchmarks

def run_backtest(pred_df: pd.DataFrame, cost_bps: float = 0.0,
                 label: str = "") -> Dict:
    """Full backtest from a prediction dataframe."""
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["strategy_return"] = df["position"] * df["target_return"]

    daily = df.groupby("date")["strategy_return"].mean().sort_index()
    to = turnover(df)
    cost = cost_bps / 1e4
    daily_net = daily - to * cost

    return {
        "label": label,
        "gross": summary(daily),
        "net":   summary(daily_net),
        "daily_gross": daily,
        "daily_net":   daily_net,
        "turnover":    to,
        "equity_gross": (1 + daily).cumprod(),
        "equity_net":   (1 + daily_net).cumprod(),
    }


# ── plotting ──────────────────────────────────────────────────────────────────
def compare_equity(model_results: list[Dict],
                   bench_results: list[Dict] = None,
                   title="Equity Curves"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # model: solid lines
    for r in model_results:
        axes[0].plot(r["equity_net"], label=r["label"])
        dd = r["equity_net"] / r["equity_net"].cummax() - 1
        axes[1].fill_between(dd.index, dd, alpha=0.3, label=r["label"])

    # benchmarks: dashed lines
    if bench_results:
        for r in bench_results:
            axes[0].plot(r["equity_net"], ls="--", label=r["label"])
            dd = r["equity_net"] / r["equity_net"].cummax() - 1
            axes[1].fill_between(dd.index, dd, alpha=0.15, label=r["label"])

    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def print_comparison(results: list[Dict]):
    rows = []
    for r in results:
        row = {"model": r["label"]}
        row.update({f"gross_{k}": v for k, v in r["gross"].items()})
        row.update({f"net_{k}": v for k, v in r["net"].items()})
        row["avg_turnover"] = float(r["turnover"].mean())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string(float_format="%.4f"))
    return df
