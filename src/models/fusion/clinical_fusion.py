"""
clinical_fusion.py
Phase 3 - Clinical Cross-Attention Fusion
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import torch
import torch.nn as nn


class ClinicalFusion(nn.Module):
    """
    Fuses clinical tabular features into image feature sequence
    using cross-attention.

    Clinical tokens  → Query
    Image features   → Key / Value

    Input:
        image_feats : [B, T, 768]   image feature sequence
        clinical    : [B, 7]        clinical feature vector
    Output:
        fused       : [B, T, fusion_dim]
    """

    def __init__(self, cfg):
        super().__init__()

        img_dim     = cfg["model"]["embed_dim"]       # 768
        fusion_dim  = cfg["model"]["fusion_dim"]      # 256
        num_heads   = cfg["model"]["num_attention_heads"]  # 8
        dropout     = cfg["model"]["dropout"]
        clinical_dim = len(cfg["data"]["clinical_features"])  # 7

        # Project clinical features to fusion_dim
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # Project image features to fusion_dim
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Cross-attention: clinical queries attend to image keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention on image sequence after fusion
        self.self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1   = nn.LayerNorm(fusion_dim)
        self.norm2   = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
        )
        self.norm3 = nn.LayerNorm(fusion_dim)

    def forward(self, image_feats, clinical):
        """
        image_feats : [B, T, 768]
        clinical    : [B, 7]
        returns     : [B, T, fusion_dim]
        """
        B, T, _ = image_feats.shape

        # Project image features → [B, T, fusion_dim]
        img = self.image_proj(image_feats)

        # Project clinical → [B, 1, fusion_dim] (single token)
        clin = self.clinical_proj(clinical)          # [B, fusion_dim]
        clin = clin.unsqueeze(1).expand(-1, T, -1)   # [B, T, fusion_dim]

        # Cross-attention: image attends to clinical context
        fused, _ = self.cross_attn(
            query=img,
            key=clin,
            value=clin,
        )
        img = self.norm1(img + self.dropout(fused))

        # Self-attention across time
        sa, _ = self.self_attn(img, img, img)
        img = self.norm2(img + self.dropout(sa))

        # Feed-forward
        img = self.norm3(img + self.dropout(self.ffn(img)))

        return img   # [B, T, fusion_dim]


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = ClinicalFusion(cfg)
    model.eval()

    image_feats = torch.randn(2, 30, 768)
    clinical    = torch.randn(2, 7)

    with torch.no_grad():
        out = model(image_feats, clinical)

    print(f"ClinicalFusion test passed")
    print(f"  image_feats : {image_feats.shape}")
    print(f"  clinical    : {clinical.shape}")
    print(f"  Output      : {out.shape}")
    print(f"  Params      : {sum(p.numel() for p in model.parameters()):,}")