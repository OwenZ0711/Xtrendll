import torch
import torch.nn as nn

from .x_trend import XTrend
from .components import TemporalBlock, CrossSectionBlock


# Extension 1: X-Trend model with Cross-Sectional Peers
class XTrendCS(XTrend):
    """
    Two refinements vs. the original design, both motivated by the
    observation that `XTrendCS` tended to under-return vs. pure `XTrend`:

    (a) **Gated residual fusion.**  The regime branch flows through the
        fusion step unmodified (`enc_y = reg_y + g ⊙ cs_proj(cs_y)`).
        `cs_proj` is zero-initialised and the gate bias is set to −2.0,
        so at step 0 the cross-section contribution is exactly zero and
        `enc_y == reg_y` — i.e. `XTrendCS` starts as vanilla `XTrend` and
        the peer branch can only be opened by gradient descent when it
        helps.  A content-aware sigmoid gate over `[reg_y, cs_y]` lets
        the model decide per-timestep how much of the peer signal to let
        in.  No second LayerNorm on `reg_y`.

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

        # Stronger dropout on the added CS branch than on the base model.
        # The CS branch is a residual on top of a competent baseline and
        # is trained on a weak signal — it overfits easily if it gets the
        # same dropout as the rest of the network.
        cs_dropout = max(self.cfg["dropout"] * 2.0, 0.2)

        self.cs_block = CrossSectionBlock(
            hid, self.cfg["num_heads"], cs_dropout
        )

        # Gated residual fusion.  `cs_proj` maps the peer representation
        # into the regime-branch space; `fuse_gate` is a content-aware
        # sigmoid over `[reg_y, cs_y]` that decides how much of the peer
        # signal to admit at each timestep.
        self.cs_proj = nn.Linear(hid, hid)
        self.fuse_gate = nn.Linear(2 * hid, hid)

        with torch.no_grad():
            # cs_proj zero-init → enc_y == reg_y at step 0 regardless of gate.
            nn.init.zeros_(self.cs_proj.weight)
            nn.init.zeros_(self.cs_proj.bias)
            # gate bias −2 → sigmoid(−2) ≈ 0.12, so even once cs_proj
            # starts moving the gate stays mostly closed until it's
            # explicitly opened.
            nn.init.constant_(self.fuse_gate.bias, -2.0)

        # (b) dedicated peer encoder (shared embedding, otherwise
        # independent of the target / key / value encoders).  Boosted
        # dropout matches the cs_block — both process the same peer
        # sequences and benefit from the same anti-overfit pressure.
        self.peer_encoder = TemporalBlock(
            input_dim, hid, num_assets,
            cs_dropout, self.emb,
        )

    def _fuse_branches(self, reg_y, cs_y):
        """Gated residual: reg_y unchanged, cs branch sigmoid-gated."""
        gate = torch.sigmoid(self.fuse_gate(torch.cat([reg_y, cs_y], dim=-1)))
        return reg_y + gate * self.cs_proj(cs_y)

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
            enc_y = self._fuse_branches(reg_y, cs_y)
        else:
            enc_y = reg_y

        dec_out = self.decoder(target_x, target_id, enc_y)
        return torch.tanh(self.head(dec_out)).squeeze(-1)
