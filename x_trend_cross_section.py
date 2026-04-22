import torch
import torch.nn as nn

from .x_trend import XTrend
from .components import TemporalBlock, CrossSectionBlock

# Extension 1: X-Trend model with Cross-Sectional Peers
class XTrendCS(XTrend):
    def __init__(self, input_dim, num_assets, cfg=None):
        super().__init__(input_dim, num_assets, cfg)
        hid = self.cfg["hidden_dim"]
        
        self.cs_block = CrossSectionBlock(
            hid, self.cfg["num_heads"], self.cfg["dropout"]
        )

        # Branch-specific projections before fusion.
        # Advised by Project Mentor
        self.reg_proj = nn.Linear(hid, hid)
        self.cs_proj = nn.Linear(hid, hid)
        self.fuse_norm = nn.LayerNorm(hid)

    def encode_peers(self, peer_x, peer_id):
        B, N, T, F = peer_x.shape
        x_flat = peer_x.reshape(B * N, T, F)
        id_flat = peer_id.reshape(B * N)
        h_flat = self.query_encoder(x_flat, id_flat)
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
