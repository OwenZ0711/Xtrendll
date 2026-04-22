"""
GP-based Change-Point Detection (Revised): 
Segments each asset's price series into stationary regimes
that serve as the context pool for episodic training.
"""
from __future__ import annotations
import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import pandas as pd

from .config import CPD
# ═══════════════════════════════════════════════════════════════════════════════
# GP primitives
# ═══════════════════════════════════════════════════════════════════════════════
def _matern32(t, sig2, ell):
    """Matérn 3/2 covariance matrix."""
    d = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3.0) * d / max(ell, 1e-6)
    return sig2 * (1.0 + r) * np.exp(-r)


def _log_marglik(y, K, noise=1e-4):
    """Log marginal likelihood via Cholesky."""
    n = len(y)
    Ky = K + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        Ky += 1e-3 * np.eye(n)
        L = np.linalg.cholesky(Ky)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(-0.5 * y @ alpha
                 - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2.0 * np.pi))


def _fit_gp(t, y):
    """Fit Matérn 3/2 GP, return (log_marglik, sig2, ell)."""
    def neg(params):
        sig2, ell = np.exp(params)
        return -_log_marglik(y, _matern32(t, sig2, ell))
    res = minimize(neg, x0=[0.0, 1.0], method="L-BFGS-B")
    sig2, ell = np.exp(res.x)
    return -res.fun, sig2, ell


# ═══════════════════════════════════════════════════════════════════════════════
# Revised Algorithm 1 from Paper
# ═══════════════════════════════════════════════════════════════════════════════
def segment_series(prices, lbw=None, nu=None, l_min=None, l_max=None):
    """Backward CPD scan on one price series. Returns [(start, end), ...]."""
    lbw   = lbw   or CPD["lbw"]
    nu    = nu    or CPD["nu"]
    l_min = l_min or CPD["l_min"]
    l_max = l_max or CPD["l_max"]

    T = len(prices)
    if T < lbw:
        return [(0, T - 1)] if T >= l_min else []

    t = T - 1
    t1 = T - 1
    regimes = []

    while t >= 0:
        w0 = max(t - lbw + 1, 0)
        w = prices[w0:t + 1].astype(np.float64)
        if len(w) < 5:
            t -= 1
            continue
        w = (w - w.mean()) / (w.std() + 1e-8)
        ti = np.arange(len(w), dtype=np.float64)

        # standard GP
        L_M, sig2, ell = _fit_gp(ti, w)

        # change-point GP: grid search over split points, reuse hyperparams
        best_L_C, best_cp = -np.inf, len(w) // 2
        for sp in range(3, len(w) - 2):
            L1 = _log_marglik(w[:sp], _matern32(ti[:sp], sig2, ell))
            L2 = _log_marglik(w[sp:], _matern32(ti[sp:] - ti[sp], sig2, ell))
            if L1 + L2 > best_L_C:
                best_L_C = L1 + L2
                best_cp = sp

        # severity: sigmoid(L_C - L_M) for numerical stability
        severity = 1.0 / (1.0 + np.exp(L_M - best_L_C))

        if severity >= nu:
            cp_abs = w0 + best_cp
            if t1 - cp_abs >= l_min:
                regimes.append((cp_abs, t1))
            t = cp_abs - 1
            t1 = t
        else:
            t -= 1
            if t1 - t > l_max:
                t = t1 - l_max
            if t1 - t == l_max:
                regimes.append((t, t1))
                t1 = t

    if t1 - max(t, 0) >= l_min:
        regimes.append((max(t, 0), t1))

    regimes.sort()
    return regimes

def _resolve_n_jobs(n_jobs):
    if n_jobs is None:
        n_jobs = os.environ.get("CPD_N_JOBS", "1")
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = int(n_jobs)
    return max(1, min(n_jobs, os.cpu_count() or 1))


def _segment_one_group(group):
    tk, prices = group
    return tk, segment_series(prices)


def segment_panel(panel, n_jobs=1, verbose=0):
    """Run CPD on every ticker. Returns {ticker: [(start, end), ...]}."""
    groups = [
        (tk, g.sort_values("date").reset_index(drop=True)["close"].values)
        for tk, g in panel.groupby("ticker", sort=False)
    ]
    n_jobs = _resolve_n_jobs(n_jobs)

    if verbose:
        print(f"Running CPD across {len(groups)} tickers with {n_jobs} worker(s) ...")

    if n_jobs == 1 or len(groups) <= 1:
        return {tk: segment_series(prices) for tk, prices in groups}

    with ProcessPoolExecutor(max_workers=min(n_jobs, len(groups))) as pool:
        return dict(pool.map(_segment_one_group, groups))


def _cache_dir(cache_dir=None):
    root = cache_dir or os.environ.get("CPD_CACHE_DIR", "./cpd_cache")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _panel_cache_key(panel, extra=""):
    digest = pd.util.hash_pandas_object(
        panel[["ticker", "date", "close"]], index=False
    ).values.tobytes()
    payload = digest + repr(sorted(CPD.items())).encode() + extra.encode()
    return hashlib.md5(payload).hexdigest()[:16]


def segment_panel_cached(panel, cache_dir=None, n_jobs=1, algo_version="v2", verbose=1):
    """Run CPD once and cache the regimes to disk for reuse across notebook runs."""
    cache_root = _cache_dir(cache_dir)
    cache_key = _panel_cache_key(panel, extra=f"segment_panel:{algo_version}")
    cache_path = cache_root / f"regimes_{cache_key}.pkl"

    if cache_path.exists():
        if verbose:
            print(f"Loading cached train regimes from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    regimes = segment_panel(panel, n_jobs=n_jobs, verbose=verbose)
    with open(cache_path, "wb") as f:
        pickle.dump(regimes, f)
    if verbose:
        print(f"Cached train regimes to {cache_path}")
    return regimes

    
# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for Rolling Context
# ═══════════════════════════════════════════════════════════════════════════════
def segment_panel_until(panel, end_date, n_jobs=1, verbose=0):
    """Run CPD using only data up to and including end_date."""
    end_date = pd.Timestamp(end_date)
    panel_cut = panel[pd.to_datetime(panel["date"]) <= end_date].copy()
    return segment_panel(panel_cut, n_jobs=n_jobs, verbose=verbose)

def build_regime_cache(panel, dates, recompute_every=21, n_jobs=1, verbose=0):
    """
    Build a causal CPD cache keyed by update date so that
    Each snapshot only uses data available up to that snapshot date.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(dates) == 0:
        return {}

    update_dates = list(dates[::recompute_every])
    if update_dates[-1] != dates[-1]:
        update_dates.append(dates[-1])

    cache = {}
    total = len(update_dates)
    for idx, dt in enumerate(update_dates, start=1):
        if verbose:
            print(f"CPD snapshot {idx}/{total}: {pd.Timestamp(dt).date()}")
        cache[pd.Timestamp(dt)] = segment_panel_until(panel, dt, n_jobs=n_jobs, verbose=0)
    return cache


def build_regime_cache_cached(panel, dates, recompute_every=21, cache_dir=None,
                              n_jobs=1, algo_version="v2", verbose=1):
    """Build a causal CPD cache once and persist it for reuse."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    cache_root = _cache_dir(cache_dir)
    extra = (
        f"regime_cache:{algo_version}:{recompute_every}:"
        + ",".join(pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates)
    )
    cache_key = _panel_cache_key(panel, extra=extra)
    cache_path = cache_root / f"regime_cache_{cache_key}.pkl"

    if cache_path.exists():
        if verbose:
            print(f"Loading cached regime snapshots from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    regime_cache = build_regime_cache(
        panel,
        dates,
        recompute_every=recompute_every,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(regime_cache, f)
    if verbose:
        print(f"Cached regime snapshots to {cache_path}")
    return regime_cache

def get_cached_regimes(regime_cache, target_date):
    """Return the latest cached regime snapshot available at or before target_date."""
    target_date = pd.Timestamp(target_date)
    if not regime_cache:
        return {}

    keys = sorted(regime_cache)
    key_ns = np.array([k.value for k in keys], dtype=np.int64)
    idx = np.searchsorted(key_ns, target_date.value, side="right") - 1
    if idx < 0:
        return {}
    return regime_cache[keys[idx]]
