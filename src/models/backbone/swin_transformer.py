"""
swin_transformer.py
Phase 3 - Backbone: Swin Transformer Tiny
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """
    Swin Transformer Tiny backbone.
    Input  : [B*T, 3, 224, 224]  (batch × slices flattened)
    Output : [B*T, 768]          (feature vector per slice)
    """

    def __init__(self, cfg):
        super().__init__()
        model_name = cfg["model"]["backbone"]
        pretrained  = cfg["model"]["pretrained"]
        dropout     = cfg["model"]["dropout"]

        # Load pretrained Swin-T from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            global_pool="avg",      # global average pool → [B, 768]
        )

        self.embed_dim = self.backbone.num_features  # 768 for Swin-T
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        returns: [B, T, 768]
        """
        B, T, C, H, W = x.shape

        # Flatten batch and time dims
        x = x.view(B * T, C, H, W)             # [B*T, 3, H, W]

        # Extract features
        feats = self.backbone(x)                # [B*T, 768]
        feats = self.norm(feats)
        feats = self.dropout(feats)

        # Restore time dimension
        feats = feats.view(B, T, self.embed_dim)  # [B, T, 768]

        return feats


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = SwinBackbone(cfg)
    model.eval()

    dummy = torch.zeros(2, 5, 3, 224, 224)   # batch=2, slices=5
    with torch.no_grad():
        out = model(dummy)

    print(f"SwinBackbone test passed")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")