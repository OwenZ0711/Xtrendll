"""
Shared building blocks (paper §3.1):
  SideInfoFFN   – Eq. 12   : fuse hidden state with asset embedding
  VSN           – Eq. 13   : variable selection network
  TemporalBlock – Eq. 14   : VSN → LSTM → skip + LayerNorm → FFN + skip
  DecoderBlock  – Eq. 19   : same idea but concatenates encoder output y_t
  Self-Attention - Eq. 17  : among contexts
  Cross-Attention - Eq. 18 : targets attending contexts
  Cross-Section : inspired by structure in https://arxiv.org/pdf/2105.10019
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


# ── Eq. 12: FFN(h_t, s) = Linear3( ELU( Linear1(h_t) + Linear2(Emb(s)) ) ) ──
class SideInfoFFN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0, emb: nn.Embedding = None):
        super().__init__()
        self.emb = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.lin_1 = nn.Linear(in_dim, hid)
        self.lin_2 = nn.Linear(hid, hid)
        self.lin_3   = nn.Linear(hid, hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid)
        if x.dim() == 3:
            e = e.unsqueeze(1)
        return self.lin_3(self.drop(F.elu(self.lin_1(x) + self.lin_2(e))))


# ── Eq. 13: per-feature FFN weighted by softmax importance ────────────────────
class VSN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0, emb: nn.Embedding = None):
        super().__init__()
        self.in_dim = in_dim
        # weight gate
        self.emb     = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.w_x     = nn.Linear(in_dim, hid)
        self.w_s     = nn.Linear(hid, hid)
        self.w_out   = nn.Linear(hid, in_dim)
        # per-feature transforms
        self.feat_fn = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hid), nn.ELU(), nn.Linear(hid, hid))
            for _ in range(in_dim)
        ])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid).unsqueeze(1)                         # [B,1,H]
        w = torch.softmax(self.w_out(F.elu(self.w_x(x) + self.w_s(e))), dim=-1)  # [B,T,F]
        parts = torch.stack([fn(x[..., j:j+1]) for j, fn in enumerate(self.feat_fn)], dim=2)  # [B,T,F,H]
        return self.norm(self.drop((w.unsqueeze(-1) * parts).sum(2)))              # [B,T,H]


# ── Eq. 14: encoder temporal block ─────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3, emb: nn.Embedding = None):
        super().__init__()
        self.emb  = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.vsn  = VSN(in_dim, hid, n_assets, dropout, self.emb)
        # per-asset LSTM init  (h0, c0) = (FFN3(Emb(s)), FFN4(Emb(s)))
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn   = SideInfoFFN(hid, hid, n_assets, dropout, self.emb)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        x0 = self.vsn(x, sid)                                 # [B,T,H]
        e  = self.emb(sid)                                     # [B,H]
        h0 = self.h0(e).unsqueeze(0)                           # [1,B,H]
        c0 = self.c0(e).unsqueeze(0)
        h, _ = self.lstm(x0, (h0, c0))                        # [B,T,H]
        a = self.norm1(h + x0)                                 # skip
        return self.norm2(self.drop(self.ffn(a, sid)) + a)     # skip


# ── Eq. 19: decoder block (fuses encoder output y_t) ──────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3, emb: nn.Embedding = None):
        super().__init__()
        self.emb  = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.vsn  = VSN(in_dim, hid, n_assets, dropout, self.emb)
        self.fuse = nn.Sequential(nn.Linear(2 * hid, hid), nn.ELU())
        self.pre_norm = nn.LayerNorm(hid)
      
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn   = SideInfoFFN(hid, hid, n_assets, dropout, self.emb)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)
      

    def forward(self, x: torch.Tensor, sid: torch.Tensor, enc_out: torch.Tensor):
        v = self.vsn(x, sid)
        f = self.pre_norm(self.fuse(torch.cat([v, enc_out], -1)))
        e = self.emb(sid)
        h0, c0 = self.h0(e).unsqueeze(0), self.c0(e).unsqueeze(0)
        h, _ = self.lstm(f, (h0, c0))
        a = self.norm1(h + f)
        return self.norm2(self.drop(self.ffn(a, sid)) + a)
      
# ── self & cross-attention ─────────
class SelfAttention(nn.Module):
    def __init__(self, hid, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)

    def forward(self, context_h):
        out, _ = self.attn(query=context_h, key=context_h, value=context_h)
        return out

class CrossAttention(nn.Module):
    def __init__(self, hid, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attn(query=query, key=key, value=value)
        return out

# Cross-Section Block (Self-Attention + Cross-Attention)
class CrossSectionBlock(nn.Module):
    """
      1. self-attend across peer assets
      2. let the target state attend to the peer set
    """
    def __init__(self, hid, num_heads=4, dropout=0.1):
        super().__init__()
        self.peer_self_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.peer_attn_norm = nn.LayerNorm(hid)
        self.peer_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.peer_ffn_norm = nn.LayerNorm(hid)

        self.target_cross_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hid)
        self.cross_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.cross_ffn_norm = nn.LayerNorm(hid)

        self.drop = nn.Dropout(dropout)

    def forward(self, target_h, peer_h, peer_mask=None):
        # target_h: [B, T, H]
        # peer_h:   [B, N, T, H]
        B, N, T, H = peer_h.shape

        # Attend across assets at each fixed time step.
        peer_tokens = peer_h.permute(0, 2, 1, 3).reshape(B * T, N, H)

        key_padding_mask = None
        if peer_mask is not None:
            key_padding_mask = ~peer_mask[:, None, :].expand(B, T, N).reshape(B * T, N)

        peer_attn, _ = self.peer_self_attn(
            peer_tokens, peer_tokens, peer_tokens,
            key_padding_mask=key_padding_mask
        )
        peer_ctx = self.peer_attn_norm(peer_tokens + self.drop(peer_attn))
        peer_ctx = self.peer_ffn_norm(peer_ctx + self.drop(self.peer_ffn(peer_ctx)))

        q = target_h.reshape(B * T, 1, H)
        cross_out, _ = self.target_cross_attn(
            q, peer_ctx, peer_ctx,
            key_padding_mask=key_padding_mask
        )
        cross_ctx = self.cross_attn_norm(q + self.drop(cross_out))
        cross_ctx = self.cross_ffn_norm(cross_ctx + self.drop(self.cross_ffn(cross_ctx)))

        return cross_ctx.reshape(B, T, H)

