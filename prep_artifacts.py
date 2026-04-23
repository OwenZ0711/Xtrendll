"""
Middle-file prep pipeline.

Runs ONCE on a local machine and produces a folder of pickles that the
Colab training notebooks can upload and load directly — no CPD, no
Bennett fit, and no feature engineering ever reruns in the cloud.

Usage
-----
  python -m S11685_Final_Project.xtrendll.prep_artifacts \\
         --universe tuning_21 \\
         --start 2008-01-01 \\
         --end   2026-04-22 \\
         --tag   v1_etf21 \\
         --cpd-jobs 8 \\
         --bennett                                            \\
         --out   ./artifacts/

Produces
--------
  artifacts/
    panel_bundle__<tag>.pkl         # panel DataFrame + feature_cols + tk2id + data_run
    train_regimes__<tag>.pkl        # cpd.segment_panel_cached output
    val_regime_cache__<tag>.pkl     # cpd.build_regime_cache_cached output for val dates
    test_regime_cache__<tag>.pkl    # same for test dates
    lead_lag_matrix__<tag>.pkl      # xtrendll.lead_lag output (only if --bennett)
    MANIFEST.json                   # configs, file list with sha256, timestamps
    _cpd_cache/                     # hidden raw cache, reused across runs

Each artefact is independent and self-describing; downstream code uses
`load_artifacts(out_dir)` to rehydrate all of them in one call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Self-contained: everything we need is a sibling module in `xtrendll`.
from .config import DATA, CPD, DEFAULT_TICKERS
from .cpd import segment_panel_cached, build_regime_cache_cached

# NOTE: `.data` imports yfinance at module load time.  Import it lazily
# inside `prep_all` so that `load_artifacts` (and therefore the top-level
# `xtrendll` package import) works without yfinance — Colab training
# notebooks only need to *load* the pickled artefacts, not re-download
# prices.
from .lead_lag import build_lead_lag_matrix_cached, build_lag_ranking_cached


# ─── universes ─────────────────────────────────────────────────────────────
# The 21-ETF tuning list mirrors XTrend_cross_section_max_peer_sweep.ipynb
# so the Bennett / lead-lag cache stays compatible with other sweeps that
# use the same universe.
TUNING_21 = [
    "SPY", "QQQ", "IWM", "VTI",
    "EFA", "EEM",
    "XLF", "XLE", "XLK", "XLI", "XLP", "XLV",
    "VNQ",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "DBC", "UUP",
]

UNIVERSES = {
    "default_50": list(DEFAULT_TICKERS),
    "tuning_21":  TUNING_21,
}


# ─── small helpers ─────────────────────────────────────────────────────────
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _dump(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── public API ────────────────────────────────────────────────────────────
def prep_all(
    universe: str,
    start: str,
    end: str,
    tag: str,
    out_dir: str,
    cpd_jobs: int = 1,
    bennett: bool = False,
    bennett_max_lag: int = 5,
    lag_rankings: bool = False,
    lag_ranking_lags: Iterable[int] = (1, 5, 10, 21, 30),
    lag_ranking_top_k: int = 5,
    lag_ranking_min_obs: int = 252,
    recompute_every: Optional[int] = None,
    verbose: int = 1,
) -> Path:
    """Build every artefact the xtrendll notebooks need; return the output dir."""
    # Lazy import: `.data` pulls in yfinance at module load.
    from .data import build_panel, time_split

    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {list(UNIVERSES)}; got {universe!r}")
    tickers = UNIVERSES[universe]

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hidden_cache = out / "_cpd_cache"
    hidden_cache.mkdir(exist_ok=True)

    if recompute_every is None:
        recompute_every = CPD["recompute_every"]

    # 1. Build the panel ---------------------------------------------------
    data_run = deepcopy(DATA)
    data_run["tickers"] = list(tickers)
    data_run["start"] = start
    data_run["end"] = end

    t0 = time.perf_counter()
    if verbose:
        print(f"[prep] build_panel ({len(tickers)} tickers, {start} → {end}) ...")
    panel, fcols, tk2id = build_panel(data_run)
    panel["date"] = pd.to_datetime(panel["date"])
    if verbose:
        print(f"[prep]   panel shape={panel.shape}, assets={len(tk2id)} "
              f"(elapsed {time.perf_counter() - t0:.1f}s)")

    # 2. Causal splits -----------------------------------------------------
    train_d, val_d, test_d = time_split(panel, data_run["train_frac"], data_run["val_frac"])
    train_end = pd.Timestamp(train_d.max())
    val_end = pd.Timestamp(val_d.max())
    test_end = pd.Timestamp(test_d.max())

    train_panel = panel[panel["date"] <= train_end].copy()
    val_panel_hist = panel[panel["date"] <= val_end].copy()
    test_panel_hist = panel[panel["date"] <= test_end].copy()

    # 3. CPD: train regimes + causal val / test snapshots ------------------
    if verbose:
        print("[prep] segment_panel_cached (train CPD) ...")
    t0 = time.perf_counter()
    train_regimes = segment_panel_cached(
        train_panel, cache_dir=str(hidden_cache), n_jobs=cpd_jobs, verbose=verbose,
    )
    if verbose:
        print(f"[prep]   regimes for {len(train_regimes)} tickers "
              f"(elapsed {time.perf_counter() - t0:.1f}s)")

    if verbose:
        print("[prep] build_regime_cache_cached (val) ...")
    t0 = time.perf_counter()
    val_regime_cache = build_regime_cache_cached(
        val_panel_hist, val_d,
        recompute_every=recompute_every,
        cache_dir=str(hidden_cache), n_jobs=cpd_jobs, verbose=verbose,
    )
    if verbose:
        print(f"[prep]   val snapshots={len(val_regime_cache)} "
              f"(elapsed {time.perf_counter() - t0:.1f}s)")

    if verbose:
        print("[prep] build_regime_cache_cached (test) ...")
    t0 = time.perf_counter()
    test_regime_cache = build_regime_cache_cached(
        test_panel_hist, test_d,
        recompute_every=recompute_every,
        cache_dir=str(hidden_cache), n_jobs=cpd_jobs, verbose=verbose,
    )
    if verbose:
        print(f"[prep]   test snapshots={len(test_regime_cache)} "
              f"(elapsed {time.perf_counter() - t0:.1f}s)")

    # 4. Optional Bennett lead-lag matrix (trained on TRAIN panel only) ----
    lead_lag_payload = None
    if bennett:
        if verbose:
            print(f"[prep] build_lead_lag_matrix_cached (max_lag={bennett_max_lag}) ...")
        t0 = time.perf_counter()
        lead_lag_payload = build_lead_lag_matrix_cached(
            train_panel, cache_dir=str(hidden_cache),
            max_lag=bennett_max_lag, verbose=verbose,
        )
        if verbose:
            print(f"[prep]   lead-lag matrix shape="
                  f"{lead_lag_payload['S'].shape} "
                  f"(elapsed {time.perf_counter() - t0:.1f}s)")

    # 4b. Optional 3D per-lag rankings (A2.5 / A3 paths) -------------------
    lag_ranking_payload = None
    if lag_rankings:
        if verbose:
            print(
                f"[prep] build_lag_ranking_cached "
                f"(lags={tuple(lag_ranking_lags)}, top_k={lag_ranking_top_k}) ..."
            )
        t0 = time.perf_counter()
        lag_ranking_payload = build_lag_ranking_cached(
            train_panel, train_d, tk2id,
            cache_dir=str(hidden_cache),
            lags=lag_ranking_lags,
            top_k=lag_ranking_top_k,
            min_obs=lag_ranking_min_obs,
            verbose=verbose,
        )
        if verbose:
            n_tk = len(lag_ranking_payload["tickers"])
            print(
                f"[prep]   lag-ranking artefact: "
                f"{len(lag_ranking_payload['lags'])} lags × {n_tk}×{n_tk} "
                f"(elapsed {time.perf_counter() - t0:.1f}s)"
            )

    # 5. Save tagged pretty copies -----------------------------------------
    panel_bundle = {
        "panel": panel,
        "fcols": list(fcols),
        "tk2id": dict(tk2id),
        "data_run": data_run,
    }
    paths = {}
    paths["panel_bundle"]      = out / f"panel_bundle__{tag}.pkl"
    paths["train_regimes"]     = out / f"train_regimes__{tag}.pkl"
    paths["val_regime_cache"]  = out / f"val_regime_cache__{tag}.pkl"
    paths["test_regime_cache"] = out / f"test_regime_cache__{tag}.pkl"

    _dump(panel_bundle,       paths["panel_bundle"])
    _dump(train_regimes,      paths["train_regimes"])
    _dump(val_regime_cache,   paths["val_regime_cache"])
    _dump(test_regime_cache,  paths["test_regime_cache"])

    if lead_lag_payload is not None:
        paths["lead_lag_matrix"] = out / f"lead_lag_matrix__{tag}.pkl"
        _dump(lead_lag_payload, paths["lead_lag_matrix"])

    if lag_ranking_payload is not None:
        paths["lag_rankings"] = out / f"lag_rankings__{tag}.pkl"
        _dump(lag_ranking_payload, paths["lag_rankings"])

    # 6. MANIFEST ----------------------------------------------------------
    manifest = {
        "tag": tag,
        "universe": universe,
        "tickers": tickers,
        "start": start,
        "end": end,
        "cpd_config": dict(CPD),
        "recompute_every": recompute_every,
        "bennett": bool(bennett),
        "bennett_max_lag": int(bennett_max_lag) if bennett else None,
        "lag_rankings": bool(lag_rankings),
        "lag_ranking_lags": list(lag_ranking_lags) if lag_rankings else None,
        "lag_ranking_top_k": int(lag_ranking_top_k) if lag_rankings else None,
        "lag_ranking_min_obs": int(lag_ranking_min_obs) if lag_rankings else None,
        "n_assets_kept": len(tk2id),
        "feature_cols": list(fcols),
        "train_days": int(len(train_d)),
        "val_days":   int(len(val_d)),
        "test_days":  int(len(test_d)),
        "git_sha": _git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            key: {"name": p.name, "sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for key, p in paths.items()
        },
    }
    with open(out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if verbose:
        print(f"\n[prep] done. Artifacts in {out}:")
        for key, p in paths.items():
            size_mb = p.stat().st_size / (1 << 20)
            print(f"  {key:<22s}  {p.name:<42s} ({size_mb:6.2f} MB)")
        print(f"  MANIFEST.json        (sha256 of each file recorded)")

    return out


# ─── loader for notebooks ──────────────────────────────────────────────────
def load_artifacts(path: str, tag: Optional[str] = None, verbose: int = 1) -> dict:
    """
    Load a prep bundle from disk.

    Returns
    -------
    dict with keys:
      panel, fcols, tk2id, data_run              (from panel_bundle)
      train_regimes, val_regime_cache,
      test_regime_cache                          (CPD artefacts)
      lead_lag_matrix (optional, A2)             (2D Bennett artefact)
      lag_rankings   (optional, A2.5 / A3)       (3D per-lag artefact)
      manifest                                   (parsed MANIFEST.json)
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"prep bundle path does not exist: {p}")

    manifest_path = p / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if tag is None and manifest is not None:
        tag = manifest.get("tag")

    if tag is None:
        # fall back to the first panel_bundle__*.pkl we can find
        hits = sorted(p.glob("panel_bundle__*.pkl"))
        if not hits:
            raise FileNotFoundError(f"no panel_bundle__*.pkl in {p}")
        tag = hits[0].stem.removeprefix("panel_bundle__")

    def _require(kind: str):
        q = p / f"{kind}__{tag}.pkl"
        if not q.exists():
            raise FileNotFoundError(f"missing {kind} pickle at {q}")
        return _load(q)

    panel_bundle = _require("panel_bundle")
    train_regimes = _require("train_regimes")
    val_regime_cache = _require("val_regime_cache")
    test_regime_cache = _require("test_regime_cache")

    ll_path = p / f"lead_lag_matrix__{tag}.pkl"
    lead_lag_matrix = _load(ll_path) if ll_path.exists() else None

    lag_rank_path = p / f"lag_rankings__{tag}.pkl"
    lag_rankings = _load(lag_rank_path) if lag_rank_path.exists() else None

    # Restore panel date dtype defensively
    panel = panel_bundle["panel"].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    if verbose:
        print(f"[load_artifacts] tag={tag}  path={p}")
        print(f"  panel rows={len(panel):,}  assets={len(panel_bundle['tk2id'])}  "
              f"features={len(panel_bundle['fcols'])}")
        print(f"  train_regimes tickers={len(train_regimes)}  "
              f"val snapshots={len(val_regime_cache)}  "
              f"test snapshots={len(test_regime_cache)}")
        if lead_lag_matrix is not None:
            print(f"  lead_lag_matrix shape={lead_lag_matrix['S'].shape}")
        if lag_rankings is not None:
            print(
                f"  lag_rankings: {len(lag_rankings['lags'])} lags × "
                f"{len(lag_rankings['tickers'])}×{len(lag_rankings['tickers'])} "
                f"top_k={lag_rankings['top_k']}"
            )

    return {
        "tag": tag,
        "panel": panel,
        "fcols": panel_bundle["fcols"],
        "tk2id": panel_bundle["tk2id"],
        "data_run": panel_bundle["data_run"],
        "train_regimes": train_regimes,
        "val_regime_cache": val_regime_cache,
        "test_regime_cache": test_regime_cache,
        "lead_lag_matrix": lead_lag_matrix,
        "lag_rankings": lag_rankings,
        "manifest": manifest,
    }


# ─── CLI entrypoint ────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m S11685_Final_Project.xtrendll.prep_artifacts",
        description="Run the one-time prep pipeline for XTrendLL notebooks.",
    )
    p.add_argument("--universe", choices=list(UNIVERSES), default="tuning_21")
    p.add_argument("--start", default="2008-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--tag", default=None,
                   help="Folder/file suffix. Default: v1_etf{N}_{start}_{end}.")
    p.add_argument("--cpd-jobs", type=int, default=max(os.cpu_count() or 1, 1))
    p.add_argument("--bennett", action="store_true",
                   help="Also build the Bennett lead-lag matrix (A2 workflow).")
    p.add_argument("--bennett-max-lag", type=int, default=5)
    p.add_argument("--lag-rankings", action="store_true",
                   help="Also build per-lag Bennett rankings (A2.5 / A3 paths).")
    p.add_argument("--lag-ranking-lags", type=str, default="1,5,10,21,30",
                   help="Comma-separated τ list for --lag-rankings.")
    p.add_argument("--lag-ranking-top-k", type=int, default=5)
    p.add_argument("--lag-ranking-min-obs", type=int, default=252)
    p.add_argument("--recompute-every", type=int, default=None)
    p.add_argument("--out", default="./artifacts")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    tag = args.tag
    if tag is None:
        n_tickers = len(UNIVERSES[args.universe])
        tag = f"v1_etf{n_tickers}_{args.start[:4]}_{args.end[:4]}"

    lag_ranking_lags = tuple(int(x) for x in args.lag_ranking_lags.split(",") if x)

    prep_all(
        universe=args.universe,
        start=args.start,
        end=args.end,
        tag=tag,
        out_dir=args.out,
        cpd_jobs=int(args.cpd_jobs),
        bennett=bool(args.bennett),
        bennett_max_lag=int(args.bennett_max_lag),
        lag_rankings=bool(args.lag_rankings),
        lag_ranking_lags=lag_ranking_lags,
        lag_ranking_top_k=int(args.lag_ranking_top_k),
        lag_ranking_min_obs=int(args.lag_ranking_min_obs),
        recompute_every=args.recompute_every,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
