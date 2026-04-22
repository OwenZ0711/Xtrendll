import torch
import torch.nn as nn

from .x_trend import XTrend
from .components import TemporalBlock, CrossSectionBlock


# Extension 1: X-Trend model with Cross-Sectional Peers
class XTrendCS(XTrend):
    """
    Two refinements vs. the original design, both motivated by the
    observation that `XTrendCS` tended to under-return vs. pure `XTrend`:

    (a) `reg_proj` is now **identity-initialised** so that at epoch 0 the
        regime branch passes through the fusion LayerNorm untouched
        (enc_y ≈ reg_y at init, exactly matching pure XTrend).  This
        prevents the first several epochs from being wasted learning a
        near-identity on reg_y from a random projection.

    (b) Peers get their **own** TemporalBlock encoder (`peer_encoder`)
        rather than sharing `query_encoder` with the target branch.
        This stops peer-branch gradients from dragging the target's own
        representation toward patterns that are only useful for peer
        attention.  Shared components are only the asset embedding
        (`self.emb`), which stays tied across all four encoders.
    """

    def __init__(self, input_dim, num_assets, cfg=None):
        super().__init__(input_dim, num_assets, cfg)
        hid = self.cfg["hidden_dim"]

        self.cs_block = CrossSectionBlock(
            hid, self.cfg["num_heads"], self.cfg["dropout"]
        )

        # Branch-specific projections before fusion.
        self.reg_proj = nn.Linear(hid, hid)
        self.cs_proj = nn.Linear(hid, hid)
        self.fuse_norm = nn.LayerNorm(hid)

        # (a) identity-init on reg_proj so the regime signal flows
        # through unchanged at step 0 — reproduces pure-XTrend behaviour
        # when cs_proj starts near zero (see XTrendLL).
        with torch.no_grad():
            nn.init.eye_(self.reg_proj.weight)
            nn.init.zeros_(self.reg_proj.bias)

        # (b) dedicated peer encoder (shared embedding, otherwise
        # independent of the target / key / value encoders).
        self.peer_encoder = TemporalBlock(
            input_dim, hid, num_assets,
            self.cfg["dropout"], self.emb,
        )

    def encode_peers(self, peer_x, peer_id):
        B, N, T, F = peer_x.shape
        x_flat = peer_x.reshape(B * N, T, F)
        id_flat = peer_id.reshape(B * N)
        h_flat = self.peer_encoder(x_flat, id_flat)          # ← separate from query_encoder
        return h_flat.reshape(B, N, T, -1)

    def forward(self, target_x, target_id, ctx_x, ctx_y, ctx_id,
                peer_x=None, peer_id=None, peer_mask=None):
        q = self.query_encoder(target_x, target_id)          # [B, T, H]

        # Historical CPD branch
        k, v = self.encode_contexts(ctx_x, ctx_y, ctx_id)
        v_prime = self.ctx_ffn(self.ctx_self_attn(v))
        reg_attn = self.cross_attn(q, k, v_prime)
        reg_y = self.post_cross_norm(self.post_cross_ffn(reg_attn))   # [B, T, H]

        # Cross-sectional peer branch
        if peer_x is not None and peer_id is not None:
            peer_h = self.encode_peers(peer_x, peer_id)               # [B, N, T, H]
            cs_y = self.cs_block(q, peer_h, peer_mask)                # [B, T, H]
            enc_y = self.fuse_norm(self.reg_proj(reg_y) + self.cs_proj(cs_y))
        else:
            enc_y = reg_y

        dec_out = self.decoder(target_x, target_id, enc_y)
        return torch.tanh(self.head(dec_out)).squeeze(-1)
