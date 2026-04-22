from __future__ import annotations
import copy, math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from tqdm.auto import tqdm

from .config import TRAIN, MODEL
# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════
def sharpe_loss_tc(pos: torch.Tensor, ret: torch.Tensor,
                warmup: int = 63, cost_bps: float = 0.0, eps: float = 1e-9) -> torch.Tensor:
    """Negative annualised Sharpe with warm-up masking."""
    if warmup > 0:
        pos, ret = pos[:, warmup:], ret[:, warmup:]
    raw = pos * ret
    turnover = torch.cat([pos[:, :1].abs(),(pos[:, 1:] - pos[:, :-1]).abs()], dim=1)
    net = raw - (cost_bps / 10_000) * turnover
    mu = net.mean()
    var = (net.pow(2).mean() - mu.pow(2)).clamp_min(eps)
    return -math.sqrt(252.0) * mu / var.sqrt()

# ═══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def annualised_sharpe(daily: pd.Series, eps=1e-9) -> float:
    x = daily.dropna().values.astype(np.float64)
    if len(x) == 0:
        return np.nan
    mu = x.mean()
    var = max((x**2).mean() - mu**2, eps)
    return float(np.sqrt(252) * mu / np.sqrt(var))


def max_drawdown(daily: pd.Series) -> float:
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())

def _daily_results_from_pred_df(pred_df: pd.DataFrame, cost_bps: float):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df["strategy_return"] = df["position"] * df["target_return"]
    df["prev_pos"] = df.groupby("ticker")["position"].shift(1).fillna(0.0)
    df["turnover"] = (df["position"] - df["prev_pos"]).abs()

    daily_gross = df.groupby("date")["strategy_return"].mean().sort_index()
    daily_turnover = df.groupby("date")["turnover"].mean().sort_index()
    daily_net = daily_gross - (cost_bps / 10_000.0) * daily_turnover

    return df, daily_gross, daily_net, daily_turnover

def panel_endpoint_sharpe_loss(pos_panel: torch.Tensor, ret_panel: torch.Tensor,
                               cost_bps: float, eps: float = 1e-9) -> torch.Tensor:
    """
    Panel-level endpoint Sharpe over consecutive dates.
    """
    if pos_panel.size(0) < 2:
        raise ValueError("panel_endpoint_sharpe_loss requires at least 2 dates per block.")

    gross = (pos_panel[1:] * ret_panel[1:]).mean(dim=1)
    turnover = (pos_panel[1:] - pos_panel[:-1]).abs().mean(dim=1)
    net = gross - (cost_bps / 10_000.0) * turnover

    mu = net.mean()
    var = (net.pow(2).mean() - mu.pow(2)).clamp_min(eps)
    return -math.sqrt(252.0) * mu / var.sqrt()


def _reshape_panel_endpoints(pos_end: torch.Tensor, ret_end: torch.Tensor,
                             dates, tickers):
    """
    Reshape flattened endpoint predictions into [dates, assets] panel matrices.
    """
    date_index = pd.to_datetime(pd.Index(dates))
    ticker_index = pd.Index(tickers)
    unique_dates = pd.Index(pd.unique(date_index))
    unique_tickers = pd.Index(pd.unique(ticker_index))

    n_dates = len(unique_dates)
    n_assets = len(unique_tickers)
    expected = n_dates * n_assets
    if pos_end.numel() != expected:
        raise ValueError(
            f"Expected {n_dates} dates x {n_assets} assets = {expected} samples, "
            f"got {pos_end.numel()}."
        )

    return pos_end.reshape(n_dates, n_assets), ret_end.reshape(n_dates, n_assets)

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Step
# ═══════════════════════════════════════════════════════════════════════════════
def _baseline_step(model, batch, device, warmup, cost_bps=None):
    if cost_bps is None:
      cost_bps = TRAIN["cost_bps"]
    x   = batch["x"].to(device)
    y   = batch["y"].to(device)       # scalar per sample
    sid = batch["sid"].to(device)
    pos = model(x, sid)               # [B,T]

    # baseline dataset gives scalar y; expand to match warm-up slicing
    # The target_return at the endpoint of each window is replicated
    # across time-steps so the Sharpe loss operates on the full sequence.
  
    loss = sharpe_loss_tc(pos, y, warmup, cost_bps=cost_bps)
    return loss, pos, y, batch["date"], batch["ticker"]

# ═══════════════════════════════════════════════════════════════════════════════
# X-trend Step
# ═══════════════════════════════════════════════════════════════════════════════
def _xtrend_step(model, batch, device, warmup, cost_bps=None):
    if cost_bps is None:
      cost_bps = TRAIN["cost_bps"]
    target_x  = batch["target_x"].to(device)       # [B, lt, F]
    target_y  = batch["target_y"].to(device)        # [B, lt]
    target_id = batch["target_id"].to(device)       # [B]
    ctx_x     = batch["ctx_x"].to(device)           # [B, C, lc, F]
    ctx_y     = batch["ctx_y"].to(device)            # [B, C, lc]
    ctx_id    = batch["ctx_id"].to(device)           # [B, C]

    pos = model(target_x, target_id, ctx_x, ctx_y, ctx_id)   # [B, lt]

    loss = sharpe_loss_tc(pos, target_y, warmup, cost_bps=cost_bps)
    return loss, pos, target_y, batch["date"], batch["ticker"]

# ═══════════════════════════════════════════════════════════════════════════════
# X-trend + Cross-Section Step
# ═══════════════════════════════════════════════════════════════════════════════
def _xtrend_cs_step(model, batch, device, warmup, cost_bps = None):
    if cost_bps is None:
      cost_bps = TRAIN["cost_bps"]
    target_x  = batch["target_x"].to(device)
    target_y  = batch["target_y"].to(device)
    target_id = batch["target_id"].to(device)
    ctx_x     = batch["ctx_x"].to(device)
    ctx_y     = batch["ctx_y"].to(device)
    ctx_id    = batch["ctx_id"].to(device)

    peer_x    = batch["peer_x"].to(device)
    peer_id   = batch["peer_id"].to(device)
    peer_mask = batch["peer_mask"].to(device)

    pos = model(target_x, target_id, ctx_x, ctx_y, ctx_id, peer_x, peer_id, peer_mask)
    loss = sharpe_loss_tc(pos, target_y, warmup, cost_bps=cost_bps)
    return loss, pos, target_y, batch["date"], batch["ticker"]



# ═══════════════════════════════════════════════════════════════════════════════
# Panel Approach for all Methods
# ═══════════════════════════════════════════════════════════════════════════════
def _baseline_step_panel(model, batch, device, warmup, cost_bps=None,
                         endpoint_weight=0.5, mag_reg=0.0):
    if cost_bps is None:
        cost_bps = TRAIN["cost_bps"]
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    sid = batch["sid"].to(device)
    pos = model(x, sid)

    intra_loss = sharpe_loss_tc(pos, y, warmup, cost_bps=cost_bps)
    pos_panel, ret_panel = _reshape_panel_endpoints(pos[:, -1], y[:, -1], batch["date"], batch["ticker"])
    end_loss = panel_endpoint_sharpe_loss(pos_panel, ret_panel, cost_bps)

    loss = (1.0 - endpoint_weight) * intra_loss + endpoint_weight * end_loss
    if mag_reg > 0.0:
        loss = loss + mag_reg * (pos.pow(2).mean() - 0.25).clamp_min(0.0)
    return loss, pos, y, batch["date"], batch["ticker"]


def _xtrend_step_panel(model, batch, device, warmup, cost_bps=None,
                       endpoint_weight=0.5, mag_reg=0.0):
    if cost_bps is None:
        cost_bps = TRAIN["cost_bps"]
    target_x = batch["target_x"].to(device)
    target_y = batch["target_y"].to(device)
    target_id = batch["target_id"].to(device)
    ctx_x = batch["ctx_x"].to(device)
    ctx_y = batch["ctx_y"].to(device)
    ctx_id = batch["ctx_id"].to(device)

    pos = model(target_x, target_id, ctx_x, ctx_y, ctx_id)
    intra_loss = sharpe_loss_tc(pos, target_y, warmup, cost_bps=cost_bps)
    pos_panel, ret_panel = _reshape_panel_endpoints(
        pos[:, -1], target_y[:, -1], batch["date"], batch["ticker"]
    )
    end_loss = panel_endpoint_sharpe_loss(pos_panel, ret_panel, cost_bps)

    loss = (1.0 - endpoint_weight) * intra_loss + endpoint_weight * end_loss
    if mag_reg > 0.0:
        loss = loss + mag_reg * (pos.pow(2).mean() - 0.25).clamp_min(0.0)
    return loss, pos, target_y, batch["date"], batch["ticker"]


def _xtrend_cs_step_panel(model, batch, device, warmup, cost_bps=None,
                          endpoint_weight=0.5, mag_reg=0.0):
    if cost_bps is None:
        cost_bps = TRAIN["cost_bps"]
    target_x = batch["target_x"].to(device)
    target_y = batch["target_y"].to(device)
    target_id = batch["target_id"].to(device)
    ctx_x = batch["ctx_x"].to(device)
    ctx_y = batch["ctx_y"].to(device)
    ctx_id = batch["ctx_id"].to(device)
    peer_x = batch["peer_x"].to(device)
    peer_id = batch["peer_id"].to(device)
    peer_mask = batch["peer_mask"].to(device)

    pos = model(target_x, target_id, ctx_x, ctx_y, ctx_id, peer_x, peer_id, peer_mask)
    intra_loss = sharpe_loss_tc(pos, target_y, warmup, cost_bps=cost_bps)
    pos_panel, ret_panel = _reshape_panel_endpoints(
        pos[:, -1], target_y[:, -1], batch["date"], batch["ticker"]
    )
    end_loss = panel_endpoint_sharpe_loss(pos_panel, ret_panel, cost_bps)

    loss = (1.0 - endpoint_weight) * intra_loss + endpoint_weight * end_loss
    if mag_reg > 0.0:
        loss = loss + mag_reg * (pos.pow(2).mean() - 0.25).clamp_min(0.0)
    return loss, pos, target_y, batch["date"], batch["ticker"]



def train_epoch(model, loader, optim, device, warmup, max_gn, step_fn, cost_bps = None, scheduler=None):
    if cost_bps is None:
        cost_bps = TRAIN["cost_bps"]
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, leave=False, desc="train"):
        loss, *_ = step_fn(model, batch, device, warmup, cost_bps)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_gn:
            nn.utils.clip_grad_norm_(model.parameters(), max_gn)
        optim.step()
        total_loss += loss.item() * batch[next(iter(batch))].size(0)
        n += batch[next(iter(batch))].size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, warmup, step_fn, cost_bps=None):
    if cost_bps is None:
        cost_bps = TRAIN["cost_bps"]
    model.eval()
    total_loss, n = 0.0, 0
    rows = []
    for batch in tqdm(loader, leave=False, desc="eval"):
        loss, pos, ret, dates, tickers = step_fn(model, batch, device, warmup, cost_bps=cost_bps)
        bs = pos.size(0)
        total_loss += loss.item() * bs
        n += bs
        # take last-step position for backtesting
        p_last = pos[:, -1].cpu().numpy() if pos.dim() == 2 else pos.cpu().numpy()
        r_last = ret[:, -1].cpu().numpy() if ret.dim() == 2 else ret.cpu().numpy()
        for i in range(bs):
            rows.append({"date": dates[i], "ticker": tickers[i],
                         "position": float(p_last[i]),
                         "target_return": float(r_last[i])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    
    df, daily_gross, daily_net, daily_turnover = _daily_results_from_pred_df(df, cost_bps)
    
    gross_sharpe = annualised_sharpe(daily_gross)
    gross_mdd = max_drawdown(daily_gross)
    net_sharpe = annualised_sharpe(daily_net)
    net_mdd = max_drawdown(daily_net)
    
    return {
        "loss": total_loss / max(n, 1),
        "pred_df": df,
        "daily_returns": daily_gross,          # keep old key for compatibility
        "daily_gross_returns": daily_gross,
        "daily_net_returns": daily_net,
        "sharpe": gross_sharpe,                # keep old key for compatibility
        "max_drawdown": gross_mdd,             # keep old key for compatibility
        "net_sharpe": net_sharpe,
        "net_max_drawdown": net_mdd,
        "avg_turnover": float(daily_turnover.mean()),
    }
  
# ═══════════════════════════════════════════════════════════════════════════════
# Actual training 
# ═══════════════════════════════════════════════════════════════════════════════
def fit(model, train_loader, val_loader, device,
        step_fn,
        tcfg = None,
        mcfg = None,
        eval_step_fn = None):
    
    if eval_step_fn is None:
      eval_step_fn = step_fn
    if tcfg is None: tcfg = TRAIN
    if mcfg is None: mcfg = MODEL
    warmup = mcfg["warmup_steps"]
    cost_bps = tcfg["cost_bps"]

    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tcfg["epochs"])

    best_state = copy.deepcopy(model.state_dict())
    best_sharpe = -float("inf")
    wait = 0
    history = []

    for ep in range(1, tcfg["epochs"] + 1):
      sampler = getattr(train_loader, "batch_sampler", None)
      if hasattr(sampler, "set_epoch"):
          sampler.set_epoch(ep)

      tl = train_epoch(
          model, train_loader, optim, device, warmup,
          tcfg["max_grad_norm"], step_fn, cost_bps=cost_bps, scheduler=scheduler
      )
      vs = eval_epoch(
          model, val_loader, device, warmup, eval_step_fn, cost_bps=cost_bps
      )

      cur_lr = scheduler.get_last_lr()[0]
      row = {
          "epoch": ep,
          "train_loss": tl,
          "val_loss": vs["loss"],
          "val_sharpe": vs["net_sharpe"],          # selection metric
          "val_sharpe_gross": vs["sharpe"],
          "val_sharpe_net": vs["net_sharpe"],
          "val_mdd": vs["net_max_drawdown"],       # selection metric
          "val_mdd_gross": vs["max_drawdown"],
          "val_mdd_net": vs["net_max_drawdown"],
          "val_turnover": vs["avg_turnover"],
          "lr": cur_lr,
      }
      history.append(row)
      print(
          f"Ep {ep:03d} | trn {tl:.4f} | val {vs['loss']:.4f} | "
          f"gross {vs['sharpe']:.4f} | net {vs['net_sharpe']:.4f} | "
          f"to {vs['avg_turnover']:.4f} | lr {cur_lr:.2e}"
      )

      if vs["net_sharpe"] > best_sharpe + 1e-4:
          best_sharpe = vs["net_sharpe"]
          best_state = copy.deepcopy(model.state_dict())
          wait = 0
      else:
          wait += 1
          if wait >= tcfg["patience"]:
              print(f"Early stop at epoch {ep}")
              break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

