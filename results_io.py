"""
Persistent storage for XTrendLL training + backtest results.

Every run writes a timestamped folder containing everything needed to
(a) reproduce the backtest tables and figures and (b) reload the trained
model later for inspection.  Structure:

    results/
        <run_tag>__<timestamp>/
            config.json              # full snapshot of MODEL + DATA + TRAIN + LL_DEFAULT + extras
            manifest.json            # git sha, hostname, torch version, seed, elapsed, best epoch
            state_dict.pt            # model weights (CPU tensors)
            history.csv              # one row per epoch — train/val losses, Sharpe, LR, ...
            predictions.csv          # eval_epoch pred_df — date, ticker, position, target_return
            backtest_summary.csv     # gross + net metrics from run_backtest
            daily_gross.csv          # daily gross returns (date, value)
            daily_net.csv            # daily net returns (date, value)
            equity_curve.png         # compare_equity figure including model + benchmarks
            comparison_table.csv     # print_comparison output

Use `save_run(...)` at the end of each notebook, and `load_run(...)` /
`compare_runs([...])` when you want to look at results later without
retraining.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch


# ─── helpers ───────────────────────────────────────────────────────────────
def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _json_safe(obj):
    """Best-effort conversion so arbitrary config snapshots round-trip."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return str(obj)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ─── save / load single run ────────────────────────────────────────────────
def save_run(
    out_root: str,
    run_tag: str,
    model: torch.nn.Module,
    history: pd.DataFrame,
    test_results: dict,
    backtest: dict,
    cfg_snapshot: dict,
    benchmarks: Optional[List[dict]] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    equity_fig=None,
    seed: Optional[int] = None,
    extras: Optional[Dict] = None,
) -> Path:
    """
    Persist every artefact from one training run.

    Parameters
    ----------
    out_root      : parent folder (typically on Google Drive).  A subfolder
                    <run_tag>__<timestamp> is created inside it.
    run_tag       : short identifier, e.g. "xtrendll_a1__v1_etf21".
    model         : trained module whose state_dict() will be saved.
    history       : DataFrame returned by train.fit().
    test_results  : dict returned by train.eval_epoch() on the test loader.
                    Must contain keys: pred_df, daily_gross_returns,
                    daily_net_returns, sharpe, net_sharpe, max_drawdown,
                    net_max_drawdown, avg_turnover, loss.
    backtest      : dict returned by backtest.run_backtest().
    cfg_snapshot  : the full MODEL + DATA + TRAIN + LL config dict to pin down
                    the run.
    benchmarks    : optional list of run_backtest dicts (EW, SPY, etc.).
    comparison_df : optional print_comparison DataFrame.
    equity_fig    : optional matplotlib.Figure from compare_equity().
    seed          : optional random seed for the run.
    extras        : optional extra scalars / serialisable info.

    Returns
    -------
    Path to the created run folder.
    """
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / f"{run_tag}__{_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # 1. config + manifest ------------------------------------------------
    with open(run_dir / "config.json", "w") as f:
        json.dump(_json_safe(cfg_snapshot), f, indent=2)

    best = None
    if isinstance(history, pd.DataFrame) and "val_sharpe" in history.columns and len(history):
        try:
            best_row = history.loc[history["val_sharpe"].idxmax()]
            best = {
                "epoch": int(best_row["epoch"]),
                "val_sharpe": float(best_row["val_sharpe"]),
                "val_sharpe_gross": float(best_row.get("val_sharpe_gross", float("nan"))),
                "val_sharpe_net": float(best_row.get("val_sharpe_net", best_row["val_sharpe"])),
                "val_mdd": float(best_row.get("val_mdd", float("nan"))),
                "val_turnover": float(best_row.get("val_turnover", float("nan"))),
            }
        except Exception:
            best = None

    manifest = {
        "run_tag": run_tag,
        "run_dir": run_dir.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
        ),
        "seed": seed,
        "best_by_val_sharpe": best,
        "test_metrics": {
            "loss":              float(test_results.get("loss", float("nan"))),
            "sharpe_gross":      float(test_results.get("sharpe", float("nan"))),
            "sharpe_net":        float(test_results.get("net_sharpe", float("nan"))),
            "mdd_gross":         float(test_results.get("max_drawdown", float("nan"))),
            "mdd_net":           float(test_results.get("net_max_drawdown", float("nan"))),
            "avg_turnover":      float(test_results.get("avg_turnover", float("nan"))),
        },
        "backtest_summary": {
            "gross": _json_safe(backtest.get("gross", {})),
            "net":   _json_safe(backtest.get("net", {})),
            "avg_turnover": float(backtest.get("turnover", pd.Series(dtype=float)).mean())
                              if "turnover" in backtest else None,
            "label": backtest.get("label", None),
        },
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "extras": _json_safe(extras or {}),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 2. weights ----------------------------------------------------------
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        run_dir / "state_dict.pt",
    )

    # 3. training history -------------------------------------------------
    if isinstance(history, pd.DataFrame):
        history.to_csv(run_dir / "history.csv", index=False)

    # 4. predictions (test) ----------------------------------------------
    pred_df = test_results.get("pred_df")
    if isinstance(pred_df, pd.DataFrame):
        pred_df.to_csv(run_dir / "predictions.csv", index=False)

    # 5. daily gross / net -----------------------------------------------
    for key, fname in (
        ("daily_gross_returns", "daily_gross.csv"),
        ("daily_net_returns",   "daily_net.csv"),
    ):
        s = test_results.get(key)
        if isinstance(s, pd.Series):
            s.rename("value").to_csv(run_dir / fname, index_label="date")

    # 6. backtest summary row ---------------------------------------------
    rows = []
    for bucket in ("gross", "net"):
        sub = backtest.get(bucket, {}) or {}
        for metric, val in sub.items():
            rows.append({"bucket": bucket, "metric": metric, "value": val})
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "backtest_summary.csv", index=False)

    # 7. comparison table (model + benchmarks) ----------------------------
    if comparison_df is not None:
        comparison_df.to_csv(run_dir / "comparison_table.csv")

    # 8. equity curve figure ---------------------------------------------
    if equity_fig is not None:
        try:
            equity_fig.savefig(run_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"[save_run] warning: could not save equity figure: {e}")

    print(f"[save_run] wrote {run_dir}")
    for p in sorted(run_dir.iterdir()):
        kb = p.stat().st_size / 1024
        print(f"  {p.name:<30s} {kb:8.1f} KB")

    return run_dir


def load_run(run_dir: str) -> Dict:
    """Rehydrate a run folder into a dict (without reconstructing the model)."""
    p = Path(run_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    def _maybe_csv(name):
        q = p / name
        return pd.read_csv(q) if q.exists() else None

    def _maybe_series(name):
        q = p / name
        if not q.exists():
            return None
        s = pd.read_csv(q, index_col=0, parse_dates=[0])
        return s.iloc[:, 0].rename(name.replace(".csv", ""))

    bundle = {
        "run_dir":            p,
        "config":             json.loads((p / "config.json").read_text())   if (p / "config.json").exists() else None,
        "manifest":           json.loads((p / "manifest.json").read_text()) if (p / "manifest.json").exists() else None,
        "history":            _maybe_csv("history.csv"),
        "predictions":        _maybe_csv("predictions.csv"),
        "backtest_summary":   _maybe_csv("backtest_summary.csv"),
        "comparison_table":   _maybe_csv("comparison_table.csv"),
        "daily_gross":        _maybe_series("daily_gross.csv"),
        "daily_net":          _maybe_series("daily_net.csv"),
        "state_dict_path":    p / "state_dict.pt" if (p / "state_dict.pt").exists() else None,
        "equity_png":         p / "equity_curve.png" if (p / "equity_curve.png").exists() else None,
    }
    return bundle


# ─── cross-run comparison ──────────────────────────────────────────────────
def compare_runs(run_dirs: Iterable[str]) -> pd.DataFrame:
    """
    Build a side-by-side table from a list of run folders.
    Returns a DataFrame indexed by run name with headline metrics from
    each run's manifest + a count of epochs.
    """
    rows = []
    for d in run_dirs:
        b = load_run(d)
        m = b["manifest"] or {}
        test = m.get("test_metrics", {}) or {}
        best = m.get("best_by_val_sharpe") or {}
        hist_rows = len(b["history"]) if isinstance(b["history"], pd.DataFrame) else 0
        rows.append({
            "run":                  Path(d).name,
            "run_tag":              m.get("run_tag"),
            "n_epochs":             hist_rows,
            "best_epoch":           best.get("epoch"),
            "best_val_sharpe":      best.get("val_sharpe"),
            "test_sharpe_gross":    test.get("sharpe_gross"),
            "test_sharpe_net":      test.get("sharpe_net"),
            "test_mdd_net":         test.get("mdd_net"),
            "test_avg_turnover":    test.get("avg_turnover"),
            "n_params":             m.get("n_parameters"),
            "git_sha":              m.get("git_sha"),
            "seed":                 m.get("seed"),
        })
    df = pd.DataFrame(rows).set_index("run")
    return df
