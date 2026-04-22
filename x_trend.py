# X-Trend Model based on Page 8 from https://arxiv.org/pdf/2310.10500

import torch, torch.nn as nn
import torch.nn.functional as F
from .components import TemporalBlock, DecoderBlock, SelfAttention, CrossAttention, SideInfoFFN
from .config import MODEL

class XTrend(nn.Module):
    def __init__(self, input_dim, num_assets, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = MODEL
        self.cfg = cfg
        hid = cfg["hidden_dim"]
      
        # Embedding
        self.emb = nn.Embedding(num_assets, hid)

        # Encoder
        self.query_encoder = TemporalBlock(input_dim, hid, num_assets, cfg["dropout"], self.emb)
        self.key_encoder   = TemporalBlock(input_dim, hid, num_assets, cfg["dropout"], self.emb)
        self.value_encoder = TemporalBlock(input_dim + 1, hid, num_assets, cfg["dropout"], self.emb)

        
        # Self-Attention
        self.ctx_self_attn = SelfAttention(hid, cfg["num_heads"], cfg["dropout"])

        # Context FFN
        self.ctx_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))

        # Cross-Attention
        self.cross_attn = CrossAttention(hid, cfg["num_heads"], cfg["dropout"])

        # Post-Attention/Pre-Decoder
        self.post_cross_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.drop = nn.Dropout(cfg["dropout"])
        self.post_cross_norm = nn.LayerNorm(hid)
        self.drop = nn.Dropout(cfg["dropout"])

        # Decoder
        self.decoder = DecoderBlock(input_dim, hid, num_assets, cfg["dropout"], self.emb)
        
        # Final Stage
        self.head = nn.Linear(hid, 1)

    def encode_contexts(self, ctx_x, ctx_y, ctx_id):
        B, C, lc, F = ctx_x.shape
        x_flat = ctx_x.reshape(B * C, lc, F)
        xy_flat = torch.cat([ctx_x, ctx_y.unsqueeze(-1)], dim=-1).reshape(B * C, lc, F + 1)
        id_flat = ctx_id.reshape(B * C)
    
        k_flat = self.key_encoder(x_flat, id_flat)[:, -1, :]
        v_flat = self.value_encoder(xy_flat, id_flat)[:, -1, :]
    
        k = k_flat.reshape(B, C, -1)
        v = v_flat.reshape(B, C, -1)
        return k, v
        
    def forward(self, target_x, target_id, ctx_x, ctx_y, ctx_id):
        q = self.query_encoder(target_x, target_id)          # [B, lt, H]
        k, v = self.encode_contexts(ctx_x, ctx_y, ctx_id)    # [B, C, H], [B, C, H]
    
        v_prime = self.ctx_ffn(self.ctx_self_attn(v))        # Eq. 17
        attn_out = self.cross_attn(q, k, v_prime)            # Att2
        y = self.post_cross_norm(self.post_cross_ffn(attn_out))  # Eq. 18
    
        dec_out = self.decoder(target_x, target_id, y)       # Eq. 19
        return torch.tanh(self.head(dec_out)).squeeze(-1)

      
      
      
      
      
        

  
      
