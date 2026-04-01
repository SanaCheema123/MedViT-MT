"""
medvit_mt.py
Phase 3 - Main Model: MedViT-MT
Multi-Task Medical Vision Transformer for GBM Brain Tumor
Tasks: Segmentation | Captioning | Grounding
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.backbone.swin_transformer import SwinBackbone
from src.models.fusion.clinical_fusion    import ClinicalFusion
from src.models.fusion.temporal_encoder   import TemporalEncoder


# ─────────────────────────────────────────────
# Task Head 1: Segmentation (UNet-style decoder)
# ─────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """
    Predicts tumor segmentation mask per slice.
    Input  : [B, T, fusion_dim]
    Output : [B, T, num_classes, H, W]
    """

    def __init__(self, fusion_dim, num_classes, image_size):
        super().__init__()
        self.image_size  = image_size
        self.num_classes = num_classes

        # Project sequence features to spatial map
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        # Decoder: upsample to full image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, seq_feats):
        """
        seq_feats : [B, T, fusion_dim]
        returns   : [B, T, num_classes, H, W]
        """
        B, T, D = seq_feats.shape

        x = self.proj(seq_feats)           # [B, T, 256]
        x = x.view(B * T, 256, 1, 1)      # [B*T, 256, 1, 1]
        x = self.decoder(x)                # [B*T, num_classes, 8, 8]

        # Upsample to full image size
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)  # [B*T, C, H, W]

        H, W = x.shape[-2:]
        x = x.view(B, T, self.num_classes, H, W)
        return x


# ─────────────────────────────────────────────
# Task Head 2: Image Captioning (Transformer decoder)
# ─────────────────────────────────────────────
class CaptioningHead(nn.Module):
    """
    Generates clinical report text from volume features.
    Input  : cls_token [B, fusion_dim]
    Output : logits    [B, seq_len, vocab_size]
    """

    def __init__(self, fusion_dim, vocab_size=1000, max_len=64, dropout=0.1):
        super().__init__()
        self.max_len    = max_len
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, fusion_dim)
        self.pos_embed   = nn.Embedding(max_len, fusion_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=8,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(fusion_dim),
        )

        self.output_proj = nn.Linear(fusion_dim, vocab_size)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, cls_token, tgt_tokens=None):
        """
        cls_token  : [B, fusion_dim]  memory from encoder
        tgt_tokens : [B, seq_len]     target token ids (teacher forcing)
        returns    : [B, seq_len, vocab_size]
        """
        B = cls_token.shape[0]
        memory = cls_token.unsqueeze(1)       # [B, 1, fusion_dim]

        if tgt_tokens is None:
            # Inference: generate dummy sequence
            seq_len = self.max_len
            tgt_tokens = torch.zeros(B, seq_len, dtype=torch.long,
                                     device=cls_token.device)
        else:
            seq_len = tgt_tokens.shape[1]

        # Embed target tokens
        positions = torch.arange(seq_len, device=cls_token.device).unsqueeze(0)
        tgt = self.token_embed(tgt_tokens) + self.pos_embed(positions)
        tgt = self.dropout(tgt)

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=cls_token.device
        )

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.output_proj(out)        # [B, seq_len, vocab_size]
        return logits


# ─────────────────────────────────────────────
# Task Head 3: Image Grounding (DETR-style)
# ─────────────────────────────────────────────
class GroundingHead(nn.Module):
    """
    Localizes lesion with bounding boxes (DETR-style).
    Input  : seq_feats [B, T, fusion_dim]
    Output : boxes     [B, T, num_queries, 4]   (cx, cy, w, h) normalized
             scores    [B, T, num_queries, 2]   (bg/fg)
    """

    def __init__(self, fusion_dim, num_queries=10, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        # Learnable object queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, fusion_dim))

        # Cross-attention: queries attend to image features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(fusion_dim)

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),             # normalize to [0,1]
        )

        # Classification head (background / lesion)
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, seq_feats):
        """
        seq_feats : [B, T, fusion_dim]
        returns   :
            boxes  [B, T, num_queries, 4]
            scores [B, T, num_queries, 2]
        """
        B, T, D = seq_feats.shape

        # Process each slice independently
        boxes_list  = []
        scores_list = []

        for t in range(T):
            feats_t = seq_feats[:, t, :].unsqueeze(1)  # [B, 1, D]
            q = self.queries.expand(B, -1, -1)          # [B, num_queries, D]

            attn_out, _ = self.cross_attn(q, feats_t, feats_t)
            q = self.norm(q + attn_out)                 # [B, num_queries, D]

            boxes  = self.bbox_head(q)                  # [B, num_queries, 4]
            scores = self.cls_head(q)                   # [B, num_queries, 2]

            boxes_list.append(boxes)
            scores_list.append(scores)

        boxes  = torch.stack(boxes_list,  dim=1)   # [B, T, num_queries, 4]
        scores = torch.stack(scores_list, dim=1)   # [B, T, num_queries, 2]

        return boxes, scores


# ─────────────────────────────────────────────
# Main Model: MedViT-MT
# ─────────────────────────────────────────────
class MedViTMT(nn.Module):
    """
    MedViT-MT: Multi-Task Medical Vision Transformer
    ─────────────────────────────────────────────
    Architecture:
      1. Swin-T Backbone       → slice features [B, T, 768]
      2. Clinical Fusion       → fused features [B, T, 256]
      3. Temporal Encoder      → seq + cls      [B, T, 256] + [B, 256]
      4a. Segmentation Head    → masks           [B, T, 2, H, W]
      4b. Captioning Head      → report logits   [B, seq, vocab]
      4c. Grounding Head       → boxes + scores  [B, T, Q, 4/2]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        fusion_dim  = cfg["model"]["fusion_dim"]
        num_classes = cfg["tasks"]["num_seg_classes"]
        num_queries = cfg["tasks"]["num_grounding_queries"]
        image_size  = cfg["data"]["image_size"]
        dropout     = cfg["model"]["dropout"]

        # Components
        self.backbone         = SwinBackbone(cfg)
        self.clinical_fusion  = ClinicalFusion(cfg)
        self.temporal_encoder = TemporalEncoder(cfg)

        # Task heads
        self.seg_head    = SegmentationHead(fusion_dim, num_classes, image_size)
        self.cap_head    = CaptioningHead(fusion_dim, dropout=dropout)
        self.ground_head = GroundingHead(fusion_dim, num_queries, dropout)

        # Classification head on CLS token (IDH1: 3 classes)
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, image_seq, clinical, valid_mask=None, tgt_tokens=None):
        """
        image_seq   : [B, T, 3, H, W]
        clinical    : [B, 7]
        valid_mask  : [B, T]  1=real, 0=pad
        tgt_tokens  : [B, seq_len]  for captioning teacher forcing

        Returns dict with all task outputs.
        """
        # 1. Backbone: extract per-slice features
        img_feats = self.backbone(image_seq)             # [B, T, 768]

        # 2. Clinical fusion
        fused = self.clinical_fusion(img_feats, clinical)  # [B, T, 256]

        # 3. Temporal encoding
        seq_out, cls_out = self.temporal_encoder(fused, valid_mask)
        # seq_out: [B, T, 256]   cls_out: [B, 256]

        # 4. Task heads
        seg_masks       = self.seg_head(seq_out)
        cap_logits      = self.cap_head(cls_out, tgt_tokens)
        boxes, scores   = self.ground_head(seq_out)
        cls_logits      = self.cls_head(cls_out)

        return {
            "seg_masks"  : seg_masks,      # [B, T, 2, H, W]
            "cap_logits" : cap_logits,      # [B, seq_len, vocab]
            "boxes"      : boxes,           # [B, T, Q, 4]
            "scores"     : scores,          # [B, T, Q, 2]
            "cls_logits" : cls_logits,      # [B, 3]
            "cls_token"  : cls_out,         # [B, 256]
        }

    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────
# Model test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("="*55)
    print("  MedViT-MT — Phase 3: Model Architecture Test")
    print("="*55)

    model = MedViTMT(cfg)
    model.eval()

    B, T = 2, 10   # small T for quick test
    image_seq  = torch.zeros(B, T, 3, 224, 224)
    clinical   = torch.randn(B, 7)
    valid_mask = torch.ones(B, T)

    print(f"\nRunning forward pass...")
    print(f"  image_seq : {image_seq.shape}")
    print(f"  clinical  : {clinical.shape}")

    with torch.no_grad():
        out = model(image_seq, clinical, valid_mask)

    print(f"\nOutputs:")
    print(f"  seg_masks  : {out['seg_masks'].shape}")
    print(f"  cap_logits : {out['cap_logits'].shape}")
    print(f"  boxes      : {out['boxes'].shape}")
    print(f"  scores     : {out['scores'].shape}")
    print(f"  cls_logits : {out['cls_logits'].shape}")

    total, trainable = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Total     : {total:,}")
    print(f"  Trainable : {trainable:,}")
    print(f"  Est. RAM  : ~{total * 4 / 1024**3:.2f} GB (fp32)")
    print("\nPhase 3 complete!")