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
import math
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


# ═════════════════════════════════════════════════════════════════════════
# Per-lag ranking artefact (A2.5 / A3 paths)
# ═════════════════════════════════════════════════════════════════════════
#
# Instead of collapsing the lag axis into a single [N, N] matrix (as
# `build_lead_lag_matrix` does), this section produces a [L, N, N] stack
# of per-lag signed correlations, a per-lag "strength" score, and a
# per-lag top-K peer mask.  Matches the schema used in the parent-dir
# `lead_lag_ranking.py` so notebooks can load either artefact.
#
# Strength = log1p(|t-stat(rho, n_obs)|), a monotone transform of
# signed Pearson correlation that rewards statistical significance.
#
# Indexing convention: matrices are [peer_id, target_id] — rows say who
# leads, columns say who is led.  The LagAwarePeerBlock then does
#     logits[b, t, v, ℓ] += α · log_strength[ℓ, v, u]
# with v = peer_id, u = target_id.


def _corr_to_strength(rho: float, n_obs: int, eps: float = 1e-12) -> float:
    """Signed Pearson → log1p(|t-stat|).

    Stays 0 for rho=0 or degenerate pairs; grows roughly linearly in |rho|
    and in sqrt(n_obs).  log1p keeps the scale compressed so a handful of
    outliers do not swamp the ranking.
    """
    if not np.isfinite(rho):
        return 0.0
    denom = max(1.0 - float(rho) ** 2, eps)
    t_abs = abs(float(rho)) * math.sqrt(max(n_obs - 2, 1) / denom)
    return float(math.log1p(t_abs))


def build_lag_ranking_artifact(
    panel: pd.DataFrame,
    train_d,
    tk2id: dict,
    lags: Iterable[int] = (1, 5, 10, 21, 30),
    top_k: int = 5,
    min_obs: int = 252,
    progress: bool = True,
) -> dict:
    """
    Train-only, per-lag peer ranking from `panel["target_return"]`.

    Parameters
    ----------
    panel     : long panel with columns at least {"date", "ticker", "target_return"}
    train_d   : an iterable of datetimes that defines the training window
                (typically `train_d` from `data.time_split`).  Only rows with
                dates in this set are used to fit the rankings — prevents
                look-ahead.
    tk2id     : {ticker: asset_id} mapping; the output is indexed in asset_id
                order so `strength[ℓ][v, u]` maps directly to model indices.
    lags      : candidate τ values.  Must match the `lag_set` you'll pass to
                `XTrendLL` so `lag_order` alignment in the notebook works.
    top_k     : number of peers kept per (target, lag) in the hard mask (A3).
    min_obs   : minimum overlapping finite observations required to compute a
                correlation.  Pairs below this threshold get strength 0 and
                fall out of the top-K.

    Returns a dict with keys matching the parent-dir `lead_lag_ranking.py`:
        lags, tickers, tk2id, top_k, min_obs, score_name,
        signed_corr : {lag: [N, N] float32},
        strength    : {lag: [N, N] float32}   (≥ 0, 0 on diagonal),
        topk_mask   : {lag: [N, N] bool}      (True for the K leaders per target),
        topk_lists  : {lag: {target_ticker: [peer_tickers...]}}  (human-readable),
        obs_count   : {lag: [N, N] int32}.
    """
    lags = tuple(int(l) for l in lags)
    if any(l < 1 for l in lags):
        raise ValueError("lags must all be >= 1")

    # tickers in asset_id order
    tickers = [tk for tk, _ in sorted(tk2id.items(), key=lambda kv: kv[1])]
    n = len(tickers)

    if "target_return" not in panel.columns:
        raise KeyError(
            "build_lag_ranking_artifact needs panel['target_return']; "
            "make sure you are passing the post-`build_panel` DataFrame."
        )

    ret = (
        panel.pivot(index="date", columns="ticker", values="target_return")
        .reindex(columns=tickers)
        .sort_index()
    )
    train_index = pd.DatetimeIndex(pd.to_datetime(list(train_d)))
    ret_train = ret.loc[ret.index.intersection(train_index)].copy()
    if ret_train.empty:
        raise ValueError("No training-date rows after pivoting panel target_return.")

    signed_corr: dict = {}
    strength: dict = {}
    topk_mask: dict = {}
    topk_lists: dict = {}
    obs_count: dict = {}

    lag_iter = tqdm(lags, desc="lag-rankings", leave=False) if progress else lags
    for lag in lag_iter:
        corr_mat = np.zeros((n, n), dtype=np.float32)
        strength_mat = np.zeros((n, n), dtype=np.float32)
        obs_mat = np.zeros((n, n), dtype=np.int32)

        for peer_id, peer_tk in enumerate(tickers):
            x = ret_train[peer_tk].shift(lag).to_numpy()
            for target_id, target_tk in enumerate(tickers):
                if peer_id == target_id:
                    continue
                y = ret_train[target_tk].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                n_obs = int(mask.sum())
                obs_mat[peer_id, target_id] = n_obs
                if n_obs < min_obs:
                    continue
                rho = np.corrcoef(x[mask], y[mask])[0, 1]
                if not np.isfinite(rho):
                    continue
                corr_mat[peer_id, target_id] = float(rho)
                strength_mat[peer_id, target_id] = _corr_to_strength(rho, n_obs)

        mask_mat = np.zeros((n, n), dtype=bool)
        topk_for_lag: dict = {}
        for target_id, target_tk in enumerate(tickers):
            order = np.argsort(-strength_mat[:, target_id])
            keep: list = []
            for peer_id in order:
                if peer_id == target_id:
                    continue
                if strength_mat[peer_id, target_id] <= 0:
                    continue
                keep.append(int(peer_id))
                if len(keep) >= top_k:
                    break
            if keep:
                mask_mat[keep, target_id] = True
            topk_for_lag[target_tk] = [tickers[i] for i in keep]

        signed_corr[lag] = corr_mat
        strength[lag] = strength_mat
        topk_mask[lag] = mask_mat
        topk_lists[lag] = topk_for_lag
        obs_count[lag] = obs_mat

    return {
        "lags":        lags,
        "tickers":     tickers,
        "tk2id":       dict(tk2id),
        "top_k":       int(top_k),
        "min_obs":     int(min_obs),
        "score_name":  "target_return_lag_tstat_log1p_v1",
        "signed_corr": signed_corr,
        "strength":    strength,
        "topk_mask":   topk_mask,
        "topk_lists":  topk_lists,
        "obs_count":   obs_count,
    }


def _lag_ranking_cache_key(
    panel: pd.DataFrame, lags: Tuple[int, ...], top_k: int,
    min_obs: int, train_d, algo_version: str,
) -> str:
    digest = pd.util.hash_pandas_object(
        panel[["ticker", "date", "target_return"]], index=False
    ).values.tobytes()
    td_hash = hashlib.md5(
        pd.to_datetime(pd.Index(train_d)).astype("int64").values.tobytes()
    ).hexdigest()[:8]
    payload = (
        digest
        + f"lag_rank:{algo_version}:{'_'.join(map(str, lags))}:".encode()
        + f"k={top_k}:min_obs={min_obs}:td={td_hash}".encode()
    )
    return hashlib.md5(payload).hexdigest()[:16]


def build_lag_ranking_cached(
    panel: pd.DataFrame,
    train_d,
    tk2id: dict,
    cache_dir=None,
    lags: Iterable[int] = (1, 5, 10, 21, 30),
    top_k: int = 5,
    min_obs: int = 252,
    algo_version: str = "v1",
    verbose: int = 1,
    progress: bool = True,
) -> dict:
    """Cached wrapper around `build_lag_ranking_artifact`."""
    root = _cache_dir(cache_dir)
    lags_t = tuple(int(l) for l in lags)
    key = _lag_ranking_cache_key(panel, lags_t, int(top_k), int(min_obs), train_d, algo_version)
    path = root / f"lag_rankings_{key}.pkl"

    if path.exists():
        if verbose:
            print(f"Loading cached lag-ranking artefact from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"Building lag-ranking artefact (lags={lags_t}, top_k={top_k}) ...")
    artefact = build_lag_ranking_artifact(
        panel, train_d, tk2id, lags=lags_t, top_k=int(top_k),
        min_obs=int(min_obs), progress=progress,
    )
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    if verbose:
        print(f"Cached lag-ranking artefact to {path}")
    return artefact


def artifact_to_lag_strength_tensor(
    artifact: dict, lag_order: Iterable[int],
) -> "np.ndarray":
    """Stack `artifact["strength"]` dict into a [L, N, N] float array in the
    order given by `lag_order`.  Returns numpy so the caller can
    `torch.from_numpy(...)` or keep on CPU.
    """
    lag_order = tuple(int(l) for l in lag_order)
    missing = [l for l in lag_order if l not in artifact["strength"]]
    if missing:
        raise KeyError(
            f"lag_order contains lags not present in artefact: {missing}. "
            f"Artefact lags = {artifact['lags']}."
        )
    return np.stack([artifact["strength"][l] for l in lag_order], axis=0).astype(np.float32)


def artifact_to_lag_topk_mask_tensor(
    artifact: dict, lag_order: Iterable[int],
) -> "np.ndarray":
    """Stack `artifact["topk_mask"]` dict into a [L, N, N] bool array."""
    lag_order = tuple(int(l) for l in lag_order)
    missing = [l for l in lag_order if l not in artifact["topk_mask"]]
    if missing:
        raise KeyError(
            f"lag_order contains lags not present in artefact: {missing}. "
            f"Artefact lags = {artifact['lags']}."
        )
    return np.stack([artifact["topk_mask"][l] for l in lag_order], axis=0).astype(bool)


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
