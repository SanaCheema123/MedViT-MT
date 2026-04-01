"""
temporal_encoder.py
Phase 3 - Temporal Transformer Encoder
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import torch
import torch.nn as nn
import math


class TemporalEncoder(nn.Module):
    """
    Transformer encoder over the MRI slice sequence.
    Captures 3D volumetric context across slices.

    Input  : [B, T, fusion_dim]   fused features per slice
    Output : [B, T, fusion_dim]   temporally-aware features
             [B, fusion_dim]      global volume token (CLS)
    """

    def __init__(self, cfg):
        super().__init__()

        fusion_dim  = cfg["model"]["fusion_dim"]          # 256
        num_heads   = cfg["model"]["num_attention_heads"] # 8
        depth       = cfg["model"]["temporal_depth"]      # 2
        dropout     = cfg["model"]["dropout"]
        max_slices  = cfg["data"]["max_slices_per_patient"]  # 30

        self.fusion_dim = fusion_dim

        # Learnable CLS token (global volume representation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings (slice position)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_slices + 1, fusion_dim)  # +1 for CLS
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(fusion_dim),
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, x, valid_mask=None):
        """
        x          : [B, T, fusion_dim]
        valid_mask : [B, T] — 1=real slice, 0=padding
        returns:
            seq_out : [B, T, fusion_dim]
            cls_out : [B, fusion_dim]
        """
        B, T, D = x.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)           # [B, T+1, D]

        # Add positional embeddings
        x = x + self.pos_embed[:, :T + 1, :]
        x = self.pos_dropout(x)

        # Build key padding mask for transformer
        # True = ignore this position
        if valid_mask is not None:
            # Prepend 1 for CLS token (always valid)
            cls_valid = torch.ones(B, 1, device=valid_mask.device)
            full_mask = torch.cat([cls_valid, valid_mask], dim=1)  # [B, T+1]
            pad_mask  = (full_mask == 0)                            # True = pad
        else:
            pad_mask = None

        # Transformer forward
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        out = self.norm(out)

        cls_out = out[:, 0, :]      # [B, fusion_dim]  global token
        seq_out = out[:, 1:, :]     # [B, T, fusion_dim]

        return seq_out, cls_out


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = TemporalEncoder(cfg)
    model.eval()

    x    = torch.randn(2, 30, 256)
    mask = torch.ones(2, 30)
    mask[0, 25:] = 0   # simulate padding

    with torch.no_grad():
        seq_out, cls_out = model(x, mask)

    print(f"TemporalEncoder test passed")
    print(f"  Input   : {x.shape}")
    print(f"  seq_out : {seq_out.shape}")
    print(f"  cls_out : {cls_out.shape}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")