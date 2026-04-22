"""
XTrendLL — X-Trend + lag-aware peer cross-section.

Extends `XTrendCS` by replacing its synchronous `CrossSectionBlock` with
`LagAwarePeerBlock`.  The new block re-uses the already-computed
`peer_h: [B, N, T, H]` by slicing it at past time offsets, so the batch
schema and data pipeline stay unchanged.

Bennett's static lead-lag adjacency S is optional.  Pass `S_matrix=A`
(where A is produced by `lead_lag.build_lead_lag_matrix_cached` and
re-ordered via `lead_lag.align_S_to_asset_ids`) together with
`ll_cfg={"use_bennett": True, ...}` to turn it on (the A2 workflow).
Leaving both defaults gives the A1 workflow (learnable τ only).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# `xtrendll` is self-contained: everything it depends on is a sibling
# module inside the package (copied from the base repo during setup).
from .x_trend_cross_section import XTrendCS
from .train import TRAIN, sharpe_loss_tc
from .lag_blocks import LagAwarePeerBlock


# ────────────────────────────────────────────────────────────────────────────
# Default block config.  Lives here, not in the global `config.py`, so that
# adding xtrendll does not touch any existing module.
# ────────────────────────────────────────────────────────────────────────────
LL_DEFAULT = {
    "lag_set":     (1, 5, 10, 21, 30),   # τ candidates, capped at 30 days
    "top_k":       3,                    # keep best 3 (peer, lag) pairs per time step
    "use_bennett": False,                # True → A2 workflow, adds Bennett prior
    "alpha_init":  0.1,                  # scale of Bennett logit bias
}


class XTrendLL(XTrendCS):
    """
    Drop-in upgrade of XTrendCS.  API parity:
        XTrendLL(input_dim, num_assets, cfg=MODEL, ll_cfg=LL_DEFAULT, S_matrix=None)

    The forward signature matches XTrendCS exactly — notebooks that already
    call XTrendCS(...) work by flipping the class name, provided peers are
    enabled on the dataset side (`include_peers=True`).

    Differences vs XTrendCS:
      * `self.cs_block` is a `LagAwarePeerBlock` instead of `CrossSectionBlock`.
      * `self.cs_proj` is re-initialised near zero so the model starts as
        vanilla X-Trend and "earns" the peer contribution during training.
    """

    def __init__(
        self,
        input_dim: int,
        num_assets: int,
        cfg=None,
        ll_cfg: Optional[dict] = None,
        S_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(input_dim, num_assets, cfg)
        hid = self.cfg["hidden_dim"]

        # Resolve LL config (user override merged on top of defaults)
        resolved = dict(LL_DEFAULT)
        if ll_cfg:
            resolved.update(ll_cfg)
        self.ll_cfg = resolved

        S_for_block = S_matrix if resolved.get("use_bennett", False) else None

        # Replace parent's cs_block with our lag-aware version.
        self.cs_block = LagAwarePeerBlock(
            hid=hid,
            num_heads=self.cfg.get("num_heads", 4),
            dropout=self.cfg.get("dropout", 0.1),
            lag_set=resolved["lag_set"],
            top_k=resolved["top_k"],
            S_matrix=S_for_block,
            alpha_init=resolved["alpha_init"],
        )

        # Near-zero init on the peer projection so the model starts ≈ XTrend
        # and has to actively open the peer branch via gradient descent.
        with torch.no_grad():
            nn.init.normal_(self.cs_proj.weight, std=1e-3)
            nn.init.zeros_(self.cs_proj.bias)

    # The only real difference in forward is that we pass target_id /
    # peer_id into cs_block for the optional Bennett lookup.
    def forward(
        self,
        target_x,
        target_id,
        ctx_x,
        ctx_y,
        ctx_id,
        peer_x=None,
        peer_id=None,
        peer_mask=None,
    ):
        q = self.query_encoder(target_x, target_id)            # [B, T, H]

        # Historical CPD branch (unchanged)
        k, v = self.encode_contexts(ctx_x, ctx_y, ctx_id)
        v_prime = self.ctx_ffn(self.ctx_self_attn(v))
        reg_attn = self.cross_attn(q, k, v_prime)
        reg_y = self.post_cross_norm(self.post_cross_ffn(reg_attn))   # [B, T, H]

        # Lag-aware peer branch
        if peer_x is not None and peer_id is not None:
            peer_h = self.encode_peers(peer_x, peer_id)               # [B, N, T, H]
            cs_y = self.cs_block(
                q, peer_h, peer_mask,
                target_id=target_id, peer_id=peer_id,
            )                                                          # [B, T, H]
            enc_y = self.fuse_norm(self.reg_proj(reg_y) + self.cs_proj(cs_y))
        else:
            enc_y = reg_y

        dec_out = self.decoder(target_x, target_id, enc_y)
        return torch.tanh(self.head(dec_out)).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
# Training step functions — thin wrappers that pull batch fields and run
# the model.  Mirror `_xtrend_cs_step` / `_xtrend_cs_step_panel` in train.py;
# lives here so notebooks import from xtrendll without any train.py edit.
#
# Two flavours are exposed:
#   * `_xtrendll_step` / `_xtrendll_step_panel` — backward-compat pure-Sharpe
#     step functions (lambda_ret = 0), so existing notebooks keep working.
#   * `make_xtrendll_step(...)` / `make_xtrendll_step_panel(...)` — factories
#     that return a step function with a non-zero `lambda_ret` baked in.
#     Use these when you want to add the net-annualised-return term to the
#     Sharpe loss (see sharpe_loss_tc docstring).
# ────────────────────────────────────────────────────────────────────────────
def make_xtrendll_step(lambda_ret: float = 0.0):
    """
    Factory returning a step function compatible with train.fit /
    train.train_epoch.  `lambda_ret` is captured by closure and forwarded
    to `sharpe_loss_tc` on every batch.

        step_fn = make_xtrendll_step(lambda_ret=1.0)
        fit(model, ..., step_fn, TRAIN_RUN, MODEL_RUN)
    """
    def _step(model, batch, device, warmup, cost_bps=None):
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

        pos = model(
            target_x, target_id, ctx_x, ctx_y, ctx_id,
            peer_x, peer_id, peer_mask,
        )
        loss = sharpe_loss_tc(
            pos, target_y, warmup,
            cost_bps=cost_bps, lambda_ret=lambda_ret,
        )
        return loss, pos, target_y, batch["date"], batch["ticker"]
    return _step


def make_xtrendll_step_panel(lambda_ret: float = 0.0,
                             endpoint_weight: float = 0.5,
                             mag_reg: float = 0.0):
    """
    Factory returning a panel-endpoint step function (same contract as
    `_xtrend_cs_step_panel`).  `lambda_ret` is forwarded to the intra-window
    Sharpe loss.
    """
    def _step(model, batch, device, warmup, cost_bps=None):
        # Local import for the panel helpers so the top-level module remains
        # safe to import in environments where only the basic training step
        # is used.
        from .train import panel_endpoint_sharpe_loss, _reshape_panel_endpoints

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

        pos = model(
            target_x, target_id, ctx_x, ctx_y, ctx_id,
            peer_x, peer_id, peer_mask,
        )
        intra_loss = sharpe_loss_tc(
            pos, target_y, warmup,
            cost_bps=cost_bps, lambda_ret=lambda_ret,
        )
        pos_panel, ret_panel = _reshape_panel_endpoints(
            pos[:, -1], target_y[:, -1], batch["date"], batch["ticker"]
        )
        end_loss = panel_endpoint_sharpe_loss(pos_panel, ret_panel, cost_bps)
        loss = (1.0 - endpoint_weight) * intra_loss + endpoint_weight * end_loss
        if mag_reg > 0.0:
            loss = loss + mag_reg * (pos.pow(2).mean() - 0.25).clamp_min(0.0)
        return loss, pos, target_y, batch["date"], batch["ticker"]
    return _step


# Backward-compat defaults (pure Sharpe, lambda_ret = 0).
_xtrendll_step       = make_xtrendll_step(lambda_ret=0.0)
_xtrendll_step_panel = make_xtrendll_step_panel(lambda_ret=0.0)
