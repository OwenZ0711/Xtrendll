"""
Lag-aware peer attention blocks.

LagAwarePeerBlock is a drop-in replacement for
`components.CrossSectionBlock` with the same (target_h, peer_h, peer_mask)
input contract.  It differs in two ways:

  1. **Lagged keys / values** — instead of scoring peer_h at time t against
     target_h at time t, the block slices peer_h at time (t - τ) for a set
     of candidate lags τ ∈ lag_set.  No extra data arrives in the batch;
     the block just re-indexes the already-encoded peer history.

  2. **Top-k sparsification** — the target at (b, t) scores N · L candidate
     (peer, lag) pairs and keeps only the top k.  This forces the model to
     pick a handful of specific leaders rather than averaging across all
     peers, and is the core of the DeltaLag design.

Optionally, a pre-computed Bennett lead-lag matrix S can be passed in.
Two shapes are accepted:

  * **2D** S ∈ R[N, N]  — A2 workflow.  Same soft bias at every lag:
        logits[b, t, v, ℓ] += α · log S[v, u]

  * **3D** S ∈ R[L, N, N] — A2.5 workflow.  Per-lag soft bias:
        logits[b, t, v, ℓ] += α · log S[ℓ, v, u]

A hard ranking mask  `rank_topk_mask ∈ {True, False}[L, N, N]`  (A3) can
additionally restrict the attention to the top-K historically best
leaders for each (target, lag); non-selected peers are forced to -inf
before the top-K softmax.

Finally, `use_delta_value=True` (A4) changes the V source from
`peer_h_lag` to `peer_h_now − peer_h_lag`, matching DeltaLag's "what
changed" value path.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagAwarePeerBlock(nn.Module):
    """
    Inputs
    ------
    target_h  : [B, T, H]      query hidden state from the target encoder
    peer_h    : [B, N, T, H]   encoded peer hidden states (unchanged vs. XTrendCS)
    peer_mask : [B, N]         True where the peer slot is valid
    target_id : [B]            (only used if Bennett S is provided)
    peer_id   : [B, N]         (only used if Bennett S is provided)

    Output
    ------
    cs_y      : [B, T, H]      same contract as CrossSectionBlock.forward
    """

    def __init__(
        self,
        hid: int,
        num_heads: int = 4,                    # kept for API parity; single-head internally
        dropout: float = 0.1,
        lag_set: Iterable[int] = (1, 5, 10, 21, 30),
        top_k: int = 3,
        S_matrix: Optional[torch.Tensor] = None,
        alpha_init: float = 0.1,
        rank_topk_mask: Optional[torch.Tensor] = None,
        use_delta_value: bool = False,
    ) -> None:
        super().__init__()
        self.hid = hid
        self.lag_set = tuple(int(x) for x in lag_set)
        if any(l < 1 for l in self.lag_set):
            raise ValueError("lag_set entries must be >= 1")
        self.L = len(self.lag_set)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        # Q / K / V projections operate on H-dim vectors; no multi-head split.
        # Tying K and V to the same linear would halve parameters but also
        # couples "which peer matters" to "what that peer's signal is" — we
        # keep them separate.
        self.W_Q = nn.Linear(hid, hid)
        self.W_K = nn.Linear(hid, hid)
        self.W_V = nn.Linear(hid, hid)

        self.drop = nn.Dropout(dropout)
        self.out_ffn = nn.Sequential(
            nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid),
        )
        self.norm = nn.LayerNorm(hid)

        # Precompute the lag indices as a non-trainable buffer so they
        # move with the model to the correct device automatically.
        self.register_buffer(
            "lag_tensor", torch.tensor(self.lag_set, dtype=torch.long),
            persistent=False,
        )

        # Bennett static prior (optional).  Accept either [N, N] or [L, N, N]
        # and remember which we got — the forward path branches on `s_ndim`.
        if S_matrix is not None:
            S = torch.as_tensor(S_matrix, dtype=torch.float32)
            if S.ndim == 2:
                if S.shape[0] != S.shape[1]:
                    raise ValueError(
                        f"2D S_matrix must be square [N,N]; got {tuple(S.shape)}"
                    )
                self.s_ndim = 2
            elif S.ndim == 3:
                if S.shape[0] != self.L:
                    raise ValueError(
                        f"3D S_matrix must have shape [L,N,N] with L={self.L}; "
                        f"got {tuple(S.shape)}"
                    )
                if S.shape[1] != S.shape[2]:
                    raise ValueError(
                        f"3D S_matrix inner dims must be square [N,N]; got {tuple(S.shape)}"
                    )
                self.s_ndim = 3
            else:
                raise ValueError(
                    f"S_matrix must be 2D [N,N] or 3D [L,N,N]; got ndim={S.ndim}"
                )
            # Store log(S) once; adding ε avoids -inf on zero entries.
            self.register_buffer("log_S", torch.log(S.clamp_min(1e-6)))
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            self.use_bennett = True
        else:
            self.register_buffer("log_S", torch.zeros(0), persistent=False)
            self.register_parameter("alpha", None)
            self.use_bennett = False
            self.s_ndim = 0

        # Hard top-K peer mask per lag (A3).  True = peer is allowed to
        # contribute; False forces that (peer, lag) logit to -inf before
        # the top-K sparsification.
        if rank_topk_mask is not None:
            M = torch.as_tensor(rank_topk_mask, dtype=torch.bool)
            if M.ndim != 3:
                raise ValueError(
                    f"rank_topk_mask must be [L,N,N]; got {tuple(M.shape)}"
                )
            if M.shape[0] != self.L or M.shape[1] != M.shape[2]:
                raise ValueError(
                    f"rank_topk_mask must have shape [L,N,N] with L={self.L}; "
                    f"got {tuple(M.shape)}"
                )
            self.register_buffer("rank_topk_mask", M)
            self.use_rank_mask = True
        else:
            self.register_buffer(
                "rank_topk_mask", torch.zeros(0, dtype=torch.bool), persistent=False,
            )
            self.use_rank_mask = False

        self.use_delta_value = bool(use_delta_value)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        target_h: torch.Tensor,
        peer_h: torch.Tensor,
        peer_mask: Optional[torch.Tensor] = None,
        target_id: Optional[torch.Tensor] = None,
        peer_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, T, H = peer_h.shape
        if target_h.shape != (B, T, H):
            raise ValueError(
                f"target_h shape {tuple(target_h.shape)} does not match "
                f"peer_h-derived (B, T, H) = ({B}, {T}, {H})"
            )
        L = self.L
        device = peer_h.device
        dtype = peer_h.dtype

        # ── 1. gather peer_h at time (t - τ) for each τ ─────────────
        # lag_tensor : [L],  t_arange : [T]
        # t_idx[l, t] = t - lag_set[l]  (negative positions are clamped and
        # later masked out with valid_time).
        lag_t = self.lag_tensor.to(device=device)
        t_arange = torch.arange(T, device=device)
        t_idx = t_arange.unsqueeze(0) - lag_t.unsqueeze(1)       # [L, T]
        valid_time = t_idx >= 0                                  # [L, T]
        t_idx_c = t_idx.clamp_min(0)                             # [L, T]

        # Advanced-index the T axis of peer_h with t_idx_c →
        # peer_h_lag[b, n, l, t, h] = peer_h[b, n, t - lag[l], h].
        peer_h_lag = peer_h[:, :, t_idx_c, :]                     # [B, N, L, T, H]
        # Permute to a scoring-friendly layout.
        peer_h_lag = peer_h_lag.permute(0, 3, 1, 2, 4).contiguous()  # [B, T, N, L, H]

        # ── 2. Q / K / V projections ────────────────────────────────
        Q = self.W_Q(target_h)                                    # [B, T, H]
        K = self.W_K(peer_h_lag)                                  # [B, T, N, L, H]

        # A4 (optional): value source is the *change* between the present
        # peer state and its τ-lagged version, matching DeltaLag's idea of
        # attending to "what moved" in the leader.  `peer_h_now` broadcasts
        # across the lag axis so subtraction is well-defined for every τ.
        if self.use_delta_value:
            peer_h_now = peer_h.unsqueeze(3).expand(-1, -1, -1, L, -1)  # [B,N,T,L,H]
            peer_h_now = peer_h_now.permute(0, 2, 1, 3, 4).contiguous() # [B,T,N,L,H]
            V_src = peer_h_now - peer_h_lag
        else:
            V_src = peer_h_lag
        V = self.W_V(V_src)                                       # [B, T, N, L, H]

        # ── 3. scaled dot-product scores ────────────────────────────
        # logits[b, t, n, l] = <Q[b,t], K[b,t,n,l]> / sqrt(H)
        logits = torch.einsum("bth,btnlh->btnl", Q, K) / math.sqrt(H)

        # ── 4. Bennett static prior (A2 / A2.5 paths) ───────────────
        if self.use_bennett and target_id is not None and peer_id is not None:
            tgt_b = target_id.unsqueeze(1).expand(-1, N)          # [B, N]
            if self.s_ndim == 2:
                # A2: shared bias across all lags.
                # log_S[v, u] → [B, N] → broadcast to [B, 1, N, 1].
                s_bias = self.log_S[peer_id, tgt_b].to(dtype)     # [B, N]
                logits = logits + self.alpha.to(dtype) * s_bias.unsqueeze(1).unsqueeze(-1)
            else:
                # A2.5: per-lag bias — gather the L slices and stack.
                # log_S has shape [L, N, N].
                prior = torch.stack(
                    [self.log_S[l_idx][peer_id, tgt_b] for l_idx in range(L)],
                    dim=-1,
                ).to(dtype)                                       # [B, N, L]
                logits = logits + self.alpha.to(dtype) * prior.unsqueeze(1)

        # ── 5. masks: causal (t - τ ≥ 0) and peer validity ─────────
        # valid_time has shape [L, T] → broadcast to [1, T, 1, L].
        mask_t = valid_time.permute(1, 0).unsqueeze(0).unsqueeze(2)   # [1, T, 1, L]
        logits = logits.masked_fill(~mask_t, float("-inf"))

        if peer_mask is not None:
            mp = peer_mask.to(torch.bool).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            logits = logits.masked_fill(~mp, float("-inf"))

        # ── 5b. A3 hard top-K peer mask (per lag) ───────────────────
        if self.use_rank_mask and target_id is not None and peer_id is not None:
            tgt_b = target_id.unsqueeze(1).expand(-1, N)          # [B, N]
            rank_mask = torch.stack(
                [self.rank_topk_mask[l_idx][peer_id, tgt_b] for l_idx in range(L)],
                dim=-1,
            )                                                     # [B, N, L]
            logits = logits.masked_fill(~rank_mask.unsqueeze(1), float("-inf"))

        # ── 6. top-k sparsification + masked softmax ────────────────
        NL = N * L
        k = min(self.top_k, NL)
        logits_flat = logits.reshape(B, T, NL)                   # [B, T, NL]
        topk_vals, topk_idx = logits_flat.topk(k, dim=-1)        # [B, T, k]

        # If every candidate at (b, t) is -inf, softmax would yield NaN.
        # Replace -inf with a large negative finite value, softmax, and
        # then multiply by 0 for the all-masked rows.
        all_inf = torch.isinf(topk_vals).all(dim=-1, keepdim=True)   # [B, T, 1]
        topk_safe = topk_vals.masked_fill(torch.isinf(topk_vals), -1e9)
        weights = F.softmax(topk_safe, dim=-1)                       # [B, T, k]
        weights = weights * (~all_inf).to(weights.dtype)
        weights = self.drop(weights)

        # ── 7. gather V at top-k picks and weighted sum ────────────
        V_flat = V.reshape(B, T, NL, H)                          # [B, T, NL, H]
        idx_h = topk_idx.unsqueeze(-1).expand(-1, -1, -1, H)      # [B, T, k, H]
        V_sel = torch.gather(V_flat, 2, idx_h)                    # [B, T, k, H]
        cs_y = (weights.unsqueeze(-1) * V_sel).sum(dim=2)         # [B, T, H]

        # ── 8. FFN + residual + LayerNorm ───────────────────────────
        cs_y = self.norm(cs_y + self.drop(self.out_ffn(cs_y)))
        return cs_y
