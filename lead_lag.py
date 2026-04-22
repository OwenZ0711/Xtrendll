"""
Bennett-style lead-lag adjacency matrix (§3.1.1 of Bennett et al. 2022).

For each ordered pair of tickers (v, u), quantify "does v lead u" by
summing the absolute Pearson cross-correlation between v's past
log-returns and u's present log-returns across a small lag window, then
form the CCF-AUC statistic

    I(v, u) = Σ_{ℓ=1..L}  |corr( r^v_{t-ℓ},  r^u_t )|

    S(v, u) = sign(I(v,u) - I(u,v)) · max(I(v,u), I(u,v))
                                     / (I(v,u) + I(u,v))

We then take the non-negative half  A[v, u] = max(S[v, u], 0)  so the
matrix has a clean "v leads u ⇒ positive weight" interpretation and is
safe to log-transform before being fed into the model as an attention
bias.  The main diagonal is zeroed.

This is deliberately the *cheapest* lead-lag extractor from the Bennett
pipeline (Pearson, no clustering).  Swapping in distance correlation or
Kendall is a one-line change in `_pair_ccf_auc` and does not affect the
caller.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover — tqdm is a soft dep
    def tqdm(x=None, **_):
        return x if x is not None else iter(())


# ─── pairwise CCF-AUC via Pearson ─────────────────────────────────────────
def _pair_ccf_auc(returns: np.ndarray, max_lag: int = 5, progress: bool = True) -> np.ndarray:
    """
    Returns I : [N, N] with I[i, j] = Σ_{ℓ=1..L} |Pearson( Y^i_{t-ℓ}, Y^j_t )|.

    returns : [T, N] float array (per-ticker log-returns, no NaN).
    """
    T, N = returns.shape
    I = np.zeros((N, N), dtype=np.float64)

    iterator = range(1, max_lag + 1)
    if progress:
        iterator = tqdm(iterator, desc="lead-lag CCF-AUC", total=max_lag, leave=False)

    for lag in iterator:
        Y_past = returns[: T - lag]           # shape [T-lag, N]  — Y^i_{t-ℓ}
        Y_now = returns[lag:]                 # shape [T-lag, N]  — Y^j_t

        # column-standardize for Pearson
        pm = Y_past.mean(axis=0, keepdims=True)
        ps = Y_past.std(axis=0, keepdims=True) + 1e-12
        nm = Y_now.mean(axis=0, keepdims=True)
        ns = Y_now.std(axis=0, keepdims=True) + 1e-12
        Y_past_n = (Y_past - pm) / ps
        Y_now_n = (Y_now - nm) / ns

        # C[i, j] = corr(Y^i_{t-lag}, Y^j_t) ≈ (Y_past_n.T @ Y_now_n) / (T - lag)
        C = (Y_past_n.T @ Y_now_n) / max(T - lag, 1)
        I += np.abs(C)

    return I


def _returns_from_panel(panel: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Pivot the long panel into a dense [T, N] log-return matrix."""
    df = panel[["date", "ticker", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    # forward-fill minor missings from holidays in one ETF vs another
    wide = wide.ffill().dropna(axis=0, how="any")
    if wide.empty:
        raise ValueError("No overlapping close observations for lead-lag fit.")
    logp = np.log(wide.values)
    returns = np.diff(logp, axis=0)
    return returns, list(wide.columns)


# ─── public builder ───────────────────────────────────────────────────────
def build_lead_lag_matrix(
    panel: pd.DataFrame,
    max_lag: int = 5,
    progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Fit Bennett's CCF-AUC lead-lag adjacency on a panel of daily closes.

    Returns
    -------
    A        : ndarray [N, N], non-negative, zero diagonal.
               A[v, u] > 0 ⇒ v historically leads u.
    tickers  : list of ticker symbols in the same order as A's rows/cols.
               The caller is responsible for mapping this order into the
               model's `asset_id` space (see xtrendll/prep_artifacts.py).
    """
    returns, tickers = _returns_from_panel(panel)
    I = _pair_ccf_auc(returns, max_lag=max_lag, progress=progress)

    # CCF-AUC statistic
    denom = I + I.T
    # Avoid 0/0 for pairs with no measurable relationship
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    sign = np.sign(I - I.T)
    S = sign * np.maximum(I, I.T) / safe_denom

    # Directed adjacency: keep only positive edges
    A = np.maximum(S, 0.0)
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32), tickers


# ─── disk cache, mirroring cpd.segment_panel_cached ───────────────────────
def _cache_dir(cache_dir=None) -> Path:
    root = cache_dir or os.environ.get("CPD_CACHE_DIR", "./cpd_cache")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lead_lag_cache_key(panel: pd.DataFrame, max_lag: int, algo_version: str) -> str:
    """
    Hash key depends only on (ticker, date, close) and the builder config.
    Not entangled with CPD config — Bennett is independent of regime
    segmentation, unlike segment_panel_cached.
    """
    digest = pd.util.hash_pandas_object(
        panel[["ticker", "date", "close"]], index=False
    ).values.tobytes()
    payload = digest + f"ll:{algo_version}:{max_lag}".encode()
    return hashlib.md5(payload).hexdigest()[:16]


def build_lead_lag_matrix_cached(
    panel: pd.DataFrame,
    cache_dir=None,
    max_lag: int = 5,
    algo_version: str = "v1",
    verbose: int = 1,
) -> dict:
    """
    Cached wrapper around `build_lead_lag_matrix`.

    Returns a dict `{"S": ndarray[N,N], "tk_order": list, "max_lag": int}`
    suitable for pickling and for passing to XTrendLL's __init__.
    """
    root = _cache_dir(cache_dir)
    key = _lead_lag_cache_key(panel, max_lag=max_lag, algo_version=algo_version)
    path = root / f"lead_lag_{key}.pkl"

    if path.exists():
        if verbose:
            print(f"Loading cached lead-lag matrix from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"Building lead-lag matrix (max_lag={max_lag}) ...")
    A, tickers = build_lead_lag_matrix(panel, max_lag=max_lag)
    payload = {"S": A, "tk_order": tickers, "max_lag": int(max_lag)}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    if verbose:
        print(f"Cached lead-lag matrix to {path}")
    return payload


# ─── helper for the model: map tk_order → tensor indexed by asset_id ──────
def align_S_to_asset_ids(payload: dict, tk2id: dict, num_assets: int) -> np.ndarray:
    """
    Bennett's matrix is indexed by position in `tk_order`.  The model
    indexes by `asset_id` from `tk2id`.  This function permutes rows and
    columns so that  S_aligned[asset_id_v, asset_id_u] = S[pos_v, pos_u].

    Pairs involving assets not present in `tk_order` get zero entries
    (they contribute no bias in the logit).
    """
    S = payload["S"]
    tk_order = payload["tk_order"]
    N = num_assets
    aligned = np.zeros((N, N), dtype=np.float32)
    for i_src, tk_i in enumerate(tk_order):
        if tk_i not in tk2id:
            continue
        ai = tk2id[tk_i]
        for j_src, tk_j in enumerate(tk_order):
            if tk_j not in tk2id:
                continue
            aj = tk2id[tk_j]
            aligned[ai, aj] = S[i_src, j_src]
    return aligned
