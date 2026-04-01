"""
losses.py
Phase 4 - Multi-Task Loss Functions
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Dice Loss (Segmentation)
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred   : [B, T, C, H, W] — raw logits
        target : [B, T, H, W]    — class indices
        """
        B, T, C, H, W = pred.shape
        pred   = pred.view(B * T, C, H, W)
        target = target.view(B * T, H, W)

        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_oh).sum(dim=(2, 3))
        union        = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


# ─────────────────────────────────────────────
# 2. Segmentation Loss (Dice + CE combined)
# ─────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice     = DiceLoss()
        self.ce       = nn.CrossEntropyLoss()
        self.dw       = dice_weight
        self.cw       = ce_weight

    def forward(self, pred, target):
        B, T, C, H, W = pred.shape
        pred_flat   = pred.view(B * T, C, H, W)
        target_flat = target.view(B * T, H, W)

        dice_loss = self.dice(pred, target)
        ce_loss   = self.ce(pred_flat, target_flat)
        return self.dw * dice_loss + self.cw * ce_loss


# ─────────────────────────────────────────────
# 3. Captioning Loss (Cross-Entropy)
# ─────────────────────────────────────────────
class CaptioningLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, targets):
        B, S, V = logits.shape
        logits = torch.clamp(logits, -10, 10)
        loss = self.ce(logits.view(B * S, V), targets.view(B * S))
        return loss if not torch.isnan(loss) else torch.tensor(0.0, requires_grad=True)


# ─────────────────────────────────────────────
# 4. Grounding Loss (GIoU + Classification)
# ─────────────────────────────────────────────
class GroundingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred_boxes, pred_scores, target_boxes=None, target_labels=None):
        """
        pred_boxes   : [B, T, Q, 4]
        pred_scores  : [B, T, Q, 2]
        target_boxes : [B, T, 4]   or None
        target_labels: [B, T]      or None
        """
        B, T, Q, _ = pred_boxes.shape

        # Classification loss on scores
        scores_flat = pred_scores.view(B * T * Q, 2)

        if target_labels is not None:
            # Binarize labels → 0=background, 1=lesion (fixes out of bounds)
            labels = (target_labels > 0).long()
            labels = labels.view(B, 1, 1).expand(B, T, Q).reshape(B * T * Q)
            cls_loss = self.cls_loss(scores_flat, labels)
        else:
            # Dummy: all background
            dummy = torch.zeros(B * T * Q, dtype=torch.long,
                                device=pred_boxes.device)
            cls_loss = self.cls_loss(scores_flat, dummy)

        # L1 box loss (when targets available)
        if target_boxes is not None:
            tgt = target_boxes.view(B * T, 1, 4).expand(-1, Q, -1)
            tgt = tgt.reshape(B * T * Q, 4)
            src = pred_boxes.view(B * T * Q, 4)
            box_loss = F.l1_loss(src, tgt)
        else:
            box_loss = torch.tensor(0.0, device=pred_boxes.device)

        return cls_loss + 0.5 * box_loss


# ─────────────────────────────────────────────
# 5. Classification Loss (IDH1)
# ─────────────────────────────────────────────
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)


# ─────────────────────────────────────────────
# 6. Combined Multi-Task Loss
# ─────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weights = cfg["training"]["task_weights"]

        self.seg_loss   = SegmentationLoss()
        self.cap_loss   = CaptioningLoss()
        self.gnd_loss   = GroundingLoss()
        self.cls_loss   = ClassificationLoss()

        self.w_seg = weights["segmentation"]   # 1.0
        self.w_cap = weights["captioning"]     # 0.5
        self.w_gnd = weights["grounding"]      # 0.5
        self.w_cls = 0.3

    def forward(self, outputs, targets):
        """
        outputs : dict from MedViTMT.forward()
        targets : dict with keys:
            seg_masks    [B, T, H, W]
            cap_tokens   [B, seq_len]
            boxes        [B, T, 4]      optional
            labels       [B]
        """
        losses = {}

        # Segmentation
        if "seg_masks" in targets:
            losses["seg"] = self.w_seg * self.seg_loss(
                outputs["seg_masks"], targets["seg_masks"]
            )

        # Captioning
        if "cap_tokens" in targets:
            losses["cap"] = self.w_cap * self.cap_loss(
                outputs["cap_logits"], targets["cap_tokens"]
            )

        # Grounding
        losses["gnd"] = self.w_gnd * self.gnd_loss(
            outputs["boxes"],
            outputs["scores"],
            targets.get("boxes", None),
            targets.get("labels", None),
        )

        # Classification
        if "labels" in targets:
            losses["cls"] = self.w_cls * self.cls_loss(
                outputs["cls_logits"], targets["labels"]
            )

        total = sum(losses.values())
        losses["total"] = total
        return losses


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    loss_fn = MultiTaskLoss(cfg)

    B, T, H, W = 2, 10, 224, 224
    outputs = {
        "seg_masks"  : torch.randn(B, T, 2, H, W),
        "cap_logits" : torch.randn(B, 64, 1000),
        "boxes"      : torch.rand(B, T, 10, 4),
        "scores"     : torch.randn(B, T, 10, 2),
        "cls_logits" : torch.randn(B, 3),
    }
    targets = {
        "seg_masks"  : torch.zeros(B, T, H, W, dtype=torch.long),
        "cap_tokens" : torch.zeros(B, 64, dtype=torch.long),
        "labels"     : torch.zeros(B, dtype=torch.long),
    }

    losses = loss_fn(outputs, targets)
    print("MultiTaskLoss test passed")
    for k, v in losses.items():
        print(f"  {k:10s}: {v.item():.4f}")