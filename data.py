from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from torch.utils.data.sampler import Sampler

from .config import (
    DATA, VOL_TARGET, VOL_LOOKBACK,
    RETURN_HORIZONS, MACD_PAIRS, EPS, DEFAULT_TICKERS, CPD
)
from .cpd import get_cached_regimes

class ConsecutiveDatePanelBatchSampler(Sampler):
    """
    Yields blocks of consecutive dates, with all assets for each date.

    Each block contains `days_per_block + 1` dates: the first date is an anchor
    for turnover, and the remaining dates contribute to the panel endpoint loss.
    """

    def __init__(self, dataset, days_per_block=8, shuffle=True, seed=42, drop_partial=True):
        self.days_per_block = days_per_block
        self.shuffle = shuffle
        self.seed = seed
        self.drop_partial = drop_partial
        self.epoch = 0

        by_date = {}
        for ds_idx, (tk, end) in enumerate(dataset.targets):
            dt = pd.Timestamp(dataset.groups[tk]["dates"][end])
            by_date.setdefault(dt, []).append((tk, ds_idx))

        ordered_dates = sorted(by_date)
        if not ordered_dates:
            self.blocks = []
            self.num_assets = 0
            return

        ref_tickers = None
        self.date_rows = []
        for dt in ordered_dates:
            row = sorted(by_date[dt], key=lambda x: x[0])
            tickers = [tk for tk, _ in row]
            if ref_tickers is None:
                ref_tickers = tickers
            elif tickers != ref_tickers:
                raise ValueError(f"Inconsistent cross-section at {dt}: {tickers} vs {ref_tickers}")
            self.date_rows.append([idx for _, idx in row])

        self.num_assets = len(ref_tickers)
        span = days_per_block + 1
        stride = max(days_per_block, 1)
        self.blocks = []

        if len(self.date_rows) >= span:
            for start in range(0, len(self.date_rows) - span + 1, stride):
                rows = self.date_rows[start:start + span]
                self.blocks.append([idx for row in rows for idx in row])

            if not drop_partial:
                tail_start = ((len(self.date_rows) - span) // stride + 1) * stride
                tail = self.date_rows[tail_start:]
                if len(tail) >= 2:
                    self.blocks.append([idx for row in tail for idx in row])

        print(
            f"ConsecutiveDatePanelBatchSampler: {len(self.blocks)} blocks, "
            f"{span} dates/block, {self.num_assets} assets/date"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            order = rng.permutation(len(self.blocks))
        else:
            order = range(len(self.blocks))
        for i in order:
            yield self.blocks[i]

    def __len__(self):
        return len(self.blocks)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
def _halflife(ts: int) -> float:
    return np.log(0.5) / np.log(1.0 - 1.0 / ts)

def _macd(close: pd.Series, S: int, L: int) -> pd.Series:
    """MACD feature normalised by rolling std (Eq. 4a-b)."""
    s_ewm = close.ewm(halflife=_halflife(S), adjust=False).mean()
    l_ewm = close.ewm(halflife=_halflife(L), adjust=False).mean()
    m = (s_ewm - l_ewm) / close.rolling(63, min_periods=63).std().clip(lower=EPS)
    return m / m.rolling(252, min_periods=252).std().clip(lower=EPS)


def build_panel(cfg=None) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Download ETF prices and build the full feature panel."""
    if cfg is None:
        cfg = DATA

    tickers = list(cfg.get("tickers", DEFAULT_TICKERS))
    prices = yf.download(
        tickers,
        start=cfg["start"],
        end=cfg["end"],
        progress=False,
        auto_adjust=False,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index().dropna(how="all")

    temp = []
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        close = close[close > EPS].copy()
        df = pd.DataFrame({"date": close.index, "ticker": ticker, "close": close.values})

        ret1 = close.pct_change()
        vol  = ret1.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK, adjust=False).std()
        scaled = ret1 * VOL_TARGET / (vol.shift(1) * np.sqrt(252)).clip(lower=EPS)

        df["daily_vol"]      = vol.values
        df["target_return"]  = scaled.shift(-1).values          # next-day vol-scaled return

        for h in RETURN_HORIZONS:
            df[f"norm_ret_{h}"] = ((close / close.shift(h) - 1) / vol.clip(lower=EPS) / np.sqrt(h)).values
        for S, L in MACD_PAIRS:
            df[f"macd_{S}_{L}"] = _macd(close, S, L).values

        temp.append(df)

    panel = pd.concat(temp, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # common start so every asset exists on every date
    common_start = panel.groupby("ticker")["date"].min().max()
    panel = panel[panel["date"] >= common_start].copy()

    feature_cols = [f"norm_ret_{h}" for h in RETURN_HORIZONS] + \
                   [f"macd_{S}_{L}" for S, L in MACD_PAIRS]
    panel = panel.dropna(subset=["target_return", "daily_vol"] + feature_cols).copy()

    tks = sorted(panel["ticker"].unique())
    tk2id = {t: i for i, t in enumerate(tks)}
    panel["asset_id"] = panel["ticker"].map(tk2id).astype(np.int64)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, feature_cols, tk2id


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Time-ordered split
# ═══════════════════════════════════════════════════════════════════════════════
def time_split(panel: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    dates = sorted(panel["date"].unique())
    n = len(dates)
    t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return (
        pd.DatetimeIndex(dates[:t1]),
        pd.DatetimeIndex(dates[t1:t2]),
        pd.DatetimeIndex(dates[t2:]),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Baseline window dataset
# ═══════════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    """Each sample: [lookback, F] features  →  scalar target_return at endpoint."""

    def __init__(self, panel: pd.DataFrame, feature_cols: Sequence[str],
                 dates: Sequence[pd.Timestamp], lookback: int = 126):
        self.fcols = list(feature_cols)
        self.lb = lookback
        dates_set = set(pd.to_datetime(dates))

        self.groups: Dict[str, dict] = {}
        self.samples: List[Tuple[str, int]] = []

        for tk, frame in panel.groupby("ticker", sort=False):
            frame = frame.sort_values("date").reset_index(drop=True)
            self.groups[tk] = {
                "x": torch.tensor(frame[self.fcols].values, dtype=torch.float32),
                "y": torch.tensor(frame["target_return"].values, dtype=torch.float32),
                "dates": pd.to_datetime(frame["date"]).tolist(),
                "asset_id": int(frame["asset_id"].iloc[0]),
            }
            for i in range(self.lb - 1, len(frame)):
                if frame["date"].iloc[i] in dates_set:
                    self.samples.append((tk, i))

        # sort by endpoint index and ticker
        self.samples.sort(key=lambda s: (self.groups[s[0]]["dates"][s[1]], s[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tk, end = self.samples[idx]
        frame = self.groups[tk]
        start = end - self.lb + 1
        return {
            "x":     frame["x"][start:end+1],                          # [T,F]
            "y":     frame["y"][start:end+1],                          # scalar
            "sid":   torch.tensor(frame["asset_id"], dtype=torch.long),
            "date":  frame["dates"][end].strftime("%Y-%m-%d"),
            "ticker": tk,
        }

def _window_collate(batch):
    return {
        "x":      torch.stack([b["x"]   for b in batch]),
        "y":      torch.stack([b["y"]   for b in batch]),
        "sid":    torch.stack([b["sid"]  for b in batch]),
        "date":   [b["date"]  for b in batch],
        "ticker": [b["ticker"] for b in batch],
    }
    
# Data Loaders
def build_baseline_loaders(panel, feature_cols, train_d, val_d, test_d, cfg = None):
    if cfg is None:
        cfg = DATA

    sets = {
        "train": WindowDataset(panel, feature_cols, train_d, lookback=cfg["lookback"]),
        "val":   WindowDataset(panel, feature_cols, val_d,   lookback=cfg["lookback"]),
        "test":  WindowDataset(panel, feature_cols, test_d,  lookback=cfg["lookback"]),
    }

    loaders = {}
    loaders = {
        "train": DataLoader(
            sets["train"],
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
        "val": DataLoader(
            sets["val"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
    }

    return sets, loaders

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  X-Trend episode dataset
# ═══════════════════════════════════════════════════════════════════════════════
class EpisodeDataset(Dataset):
    """
    Each sample:
      target  – [l_t, F] features + [l_t] returns
      context – [C, l_c, F] features + [C, l_c] returns  (regime-aligned via CPD)

    Train:
      - uses a single regime set built from train history only
      - samples contexts freely from that train context pool

    Val/Test:
      - uses a causal regime cache keyed by date
      - each target date only sees contexts available strictly before that date
    """

    def __init__(self, panel, feature_cols, target_dates, ctx_pool_dates,
                 regimes=None, regime_cache=None,
                 target_len=126, ctx_len=21, num_ctx=10,
                 mode="train", seed=42,
                 include_peers=False, max_peers=None):
        assert mode in ("train", "val", "test")
        if mode == "train" and regimes is None:
            raise ValueError("EpisodeDataset(train) requires `regimes`.")
        if mode in ("val", "test") and regime_cache is None:
            raise ValueError(f"EpisodeDataset({mode}) requires `regime_cache`.")

        self.mode, self.seed = mode, seed
        self.fcols = list(feature_cols)
        self.tl, self.cl, self.nc = target_len, ctx_len, num_ctx

        # Cross-sectional Peers Setup
        self.include_peers = include_peers
        self.max_peers = max_peers
        self.num_assets = len(panel["ticker"].unique())
        self.n_peer_slots = max(max_peers if max_peers is not None else self.num_assets - 1, 1)
        self.all_window_targets = []
        self.peer_targets_by_date = {}

        tgt_set = set(pd.to_datetime(target_dates))
        ctx_set = set(pd.to_datetime(ctx_pool_dates))

        self.groups = {}
        self.targets = []
        self.ctx_pool = []
        self.ctx_pool_by_date = {}
        self.original_num_targets = 0
        self.dropped_targets = 0

        for tk, g in panel.groupby("ticker", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            rec = {
                "x": torch.tensor(g[self.fcols].values, dtype=torch.float32),
                "y": torch.tensor(g["target_return"].values, dtype=torch.float32),
                "dates": pd.to_datetime(g["date"]).tolist(),
                "asset_id": int(g["asset_id"].iloc[0]),
            }
            self.groups[tk] = rec

            for i in range(self.tl - 1, len(g)):
                if g["date"].iloc[i] in tgt_set:
                    self.targets.append((tk, i))

        self.targets.sort(key=lambda s: (self.groups[s[0]]["dates"][s[1]], s[0]))
        self.original_num_targets = len(self.targets)

        self.all_window_targets = list(self.targets)
                     
        # Cross-Sectional Peers
        if self.include_peers:
            for tk, end in self.all_window_targets:
                tgt_ts = pd.Timestamp(self.groups[tk]["dates"][end])
                self.peer_targets_by_date.setdefault(tgt_ts, []).append((tk, end))
            for tgt_ts in self.peer_targets_by_date:
                self.peer_targets_by_date[tgt_ts].sort(key=lambda z: z[0])

        if self.mode == "train":
            self.ctx_pool = self._build_ctx_pool(regimes, ctx_set)
            if len(self.ctx_pool) == 0:
                raise ValueError("Training context pool is empty. Check CPD segmentation and train dates.")
        else:
            unique_target_dates = sorted({self.groups[tk]["dates"][end] for tk, end in self.targets})

            for tgt_date in unique_target_dates:
                tgt_ts = pd.Timestamp(tgt_date)
                regimes_t = get_cached_regimes(regime_cache, tgt_ts)
                self.ctx_pool_by_date[tgt_ts] = self._build_ctx_pool(
                    regimes_t, ctx_set, tgt_date=tgt_ts
                )

            kept_targets = []
            for tk, end in self.targets:
                tgt_ts = pd.Timestamp(self.groups[tk]["dates"][end])
                if len(self.ctx_pool_by_date.get(tgt_ts, [])) > 0:
                    kept_targets.append((tk, end))

            self.targets = kept_targets
            self.dropped_targets = self.original_num_targets - len(self.targets)

    def __len__(self):
        return len(self.targets)

    def _build_ctx_pool(self, regimes, ctx_set, tgt_date=None):
        pool = []
        tgt_ts = pd.Timestamp(tgt_date) if tgt_date is not None else None

        for tk, intervals in regimes.items():
            if tk not in self.groups:
                continue
            rec = self.groups[tk]

            for rs, re in intervals:
                end_date = pd.Timestamp(rec["dates"][re])
                if end_date not in ctx_set:
                    continue
                if tgt_ts is not None and end_date >= tgt_ts:
                    continue
                pool.append((tk, rs, re, end_date))

        pool.sort(key=lambda s: (s[3], s[0]))
        return pool

    def _pick_ctx_entries(self, tgt_date, idx):
        if self.mode == "train":
            pool = self.ctx_pool
            if len(pool) == 0:
                raise IndexError("Training context pool is empty.")
            rng = np.random.default_rng(self.seed + idx)
            picks = rng.choice(len(pool), self.nc, replace=(len(pool) < self.nc))
        else:
            pool = self.ctx_pool_by_date[pd.Timestamp(tgt_date)]
            if len(pool) == 0:
                raise IndexError(f"No causal contexts available for target date {tgt_date}.")
            rng = np.random.default_rng(self.seed + idx)
            picks = rng.choice(len(pool), self.nc, replace=(len(pool) < self.nc))
    
        picks = np.atleast_1d(picks).tolist()
        return [pool[p] for p in picks]

    def _slice_ctx(self, tk, rs, re):
        """Extract one context window, pad/truncate to l_c."""
        cg = self.groups[tk]
        regime_len = re - rs + 1
        if regime_len >= self.cl:
            cx = cg["x"][re - self.cl + 1:re + 1]
            cy = cg["y"][re - self.cl + 1:re + 1]
        else:
            pad = self.cl - regime_len
            cx = torch.cat([torch.zeros(pad, cg["x"].size(1)), cg["x"][rs:re + 1]])
            cy = torch.cat([torch.zeros(pad), cg["y"][rs:re + 1]])
        return cx, cy, cg["asset_id"]

    def _get_peer_entries(self, target_tk, tgt_date, idx):
        peers = []
        for ptk, pend in self.peer_targets_by_date.get(pd.Timestamp(tgt_date), []):
            if ptk == target_tk:
                continue
            peers.append((ptk, pend, self.groups[ptk]["asset_id"]))
    
        if self.max_peers is not None and len(peers) > self.max_peers:
            if self.mode == "train":
                rng = np.random.default_rng(self.seed + idx * 7919 + 1)
                picks = rng.choice(len(peers), self.max_peers, replace=False)
                picks = sorted(np.atleast_1d(picks).tolist())
                peers = [peers[i] for i in picks]
            else:
                peers = peers[:self.max_peers]
    
        return peers

    def _slice_peer_window(self, tk, end):
        g = self.groups[tk]
        s = end - self.tl + 1
        return g["x"][s:end + 1], g["asset_id"]


    def _pack_peers(self, peer_entries):
        F = len(self.fcols)
        peer_x = torch.zeros(self.n_peer_slots, self.tl, F)
        peer_id = torch.zeros(self.n_peer_slots, dtype=torch.long)
        peer_mask = torch.zeros(self.n_peer_slots, dtype=torch.bool)
    
        for j, (ptk, pend, paid) in enumerate(peer_entries[:self.n_peer_slots]):
            x, aid = self._slice_peer_window(ptk, pend)
            peer_x[j] = x
            peer_id[j] = aid
            peer_mask[j] = True
    
        return peer_x, peer_id, peer_mask
    

    def __getitem__(self, idx):
        tk, end = self.targets[idx]
        g = self.groups[tk]
        s = end - self.tl + 1
        tgt_date = g["dates"][end]

        entries = self._pick_ctx_entries(tgt_date, idx)
        cx, cy, cid = [], [], []
        for ctk, rs, re, _ in entries:
            x, y, aid = self._slice_ctx(ctk, rs, re)
            cx.append(x)
            cy.append(y)
            cid.append(aid)

        item = {
            "target_x":  g["x"][s:end + 1],
            "target_y":  g["y"][s:end + 1],
            "target_id": torch.tensor(g["asset_id"], dtype=torch.long),
            "ctx_x":     torch.stack(cx),
            "ctx_y":     torch.stack(cy),
            "ctx_id":    torch.tensor(cid, dtype=torch.long),
            "date":      tgt_date.strftime("%Y-%m-%d"),
            "ticker":    tk,
        }
        
        if self.include_peers:
            peer_entries = self._get_peer_entries(tk, tgt_date, idx)
            peer_x, peer_id, peer_mask = self._pack_peers(peer_entries)
            item["peer_x"] = peer_x
            item["peer_id"] = peer_id
            item["peer_mask"] = peer_mask
        
        return item

def _episode_collate(batch):
    out = {
        "target_x":  torch.stack([b["target_x"]  for b in batch]),
        "target_y":  torch.stack([b["target_y"]  for b in batch]),
        "target_id": torch.stack([b["target_id"] for b in batch]),
        "ctx_x":     torch.stack([b["ctx_x"]     for b in batch]),
        "ctx_y":     torch.stack([b["ctx_y"]     for b in batch]),
        "ctx_id":    torch.stack([b["ctx_id"]    for b in batch]),
        "date":      [b["date"]   for b in batch],
        "ticker":    [b["ticker"] for b in batch],
    }

    if "peer_x" in batch[0]:
        out["peer_x"] = torch.stack([b["peer_x"] for b in batch])
        out["peer_id"] = torch.stack([b["peer_id"] for b in batch])
        out["peer_mask"] = torch.stack([b["peer_mask"] for b in batch])

    return out

def build_episode_loaders(panel, feature_cols, train_d, val_d, test_d,
                          train_regimes, cfg=None, regime_caches=None,
                          include_peers=False, max_peers=None,
                          panel_turnover=False, panel_block_days=8):
    if cfg is None:
        cfg = DATA
    if regime_caches is None:
        regime_caches = {}

    all_train_val = train_d.append(val_d) if hasattr(train_d, "append") \
                    else train_d.union(val_d)

    sets = {
        "train": EpisodeDataset(
            panel, feature_cols,
            target_dates=train_d, ctx_pool_dates=train_d,
            regimes=train_regimes,
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="train", seed=cfg["seed"], include_peers=include_peers,
            max_peers=max_peers,
        ),
        "val": EpisodeDataset(
            panel, feature_cols,
            target_dates=val_d, ctx_pool_dates=all_train_val,
            regime_cache=regime_caches.get("val"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="val", seed=cfg["seed"], include_peers=include_peers,
            max_peers=max_peers,
        ),
        "test": EpisodeDataset(
            panel, feature_cols,
            target_dates=test_d, ctx_pool_dates=all_train_val.union(test_d),
            regime_cache=regime_caches.get("test"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="test", seed=cfg["seed"], include_peers=include_peers,
            max_peers=max_peers,
        ),
    }

    for split, ds in sets.items():
        print(
            f"{split}: kept {len(ds):,}/{ds.original_num_targets:,} targets "
            f"({ds.dropped_targets:,} dropped for no causal contexts)"
        )
        if split != "train":
            avg_pool = np.mean([len(v) for v in ds.ctx_pool_by_date.values()]) if ds.ctx_pool_by_date else 0.0
            print(f"  avg causal context pool per target date: {avg_pool:.1f}")

    if panel_turnover:
        train_batch_sampler = ConsecutiveDatePanelBatchSampler(
            sets["train"],
            days_per_block=panel_block_days,
            shuffle=True,
            seed=cfg["seed"],
        )
        train_loader = DataLoader(
            sets["train"],
            batch_sampler=train_batch_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        )
    else:
        train_loader = DataLoader(
            sets["train"],
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        )
    
    loaders = {
        "train": train_loader,
        "val": DataLoader(
            sets["val"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        ),
    }

    return sets, loaders
