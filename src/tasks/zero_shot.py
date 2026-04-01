"""
zero_shot.py
Phase 5 - Zero-Shot Learning Module
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
CLIP-style semantic embedding approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.medvit_mt import MedViTMT
from src.data.dataset     import GBMDataset


# ─────────────────────────────────────────────
# 1. Class Text Descriptions (GBM domain)
# ─────────────────────────────────────────────
CLASS_DESCRIPTIONS = {
    0: [  # IDH1 Wildtype
        "glioblastoma wildtype aggressive tumor high grade",
        "IDH wildtype glioma poor prognosis rapid progression",
        "wildtype GBM high proliferation necrosis enhancement",
    ],
    1: [  # IDH1 Mutated
        "IDH mutated glioma better prognosis lower grade",
        "mutated IDH1 glioma slower progression younger patient",
        "IDH mutation glioma favorable outcome longer survival",
    ],
    2: [  # NOS/NEC
        "glioma not otherwise specified unclassified tumor",
        "NOS NEC glioma indeterminate classification uncertain",
        "unspecified glioma mixed features inconclusive markers",
    ],
}


# ─────────────────────────────────────────────
# 2. Simple Text Encoder (bag-of-words style)
# ─────────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Lightweight text encoder using fixed vocabulary embeddings.
    Maps clinical text descriptions to embedding space.
    """

    def __init__(self, embed_dim=256, vocab_size=500):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def text_to_ids(self, text, max_len=20):
        """Simple character-hash tokenizer (no external tokenizer needed)."""
        words  = text.lower().split()[:max_len]
        ids    = [abs(hash(w)) % 499 + 1 for w in words]
        # Pad to max_len
        ids   += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, texts):
        """
        texts  : list of strings
        returns: [N, embed_dim]
        """
        ids    = torch.stack([self.text_to_ids(t) for t in texts])
        embeds = self.embed(ids)             # [N, max_len, embed_dim]
        pooled = embeds.mean(dim=1)          # [N, embed_dim]
        return F.normalize(self.proj(pooled), dim=-1)


# ─────────────────────────────────────────────
# 3. Zero-Shot Classifier
# ─────────────────────────────────────────────
class ZeroShotClassifier(nn.Module):
    """
    CLIP-style zero-shot:
    1. Encode class descriptions with TextEncoder
    2. Encode images with MedViTMT CLS token
    3. Classify by cosine similarity
    """

    def __init__(self, cfg):
        super().__init__()
        embed_dim       = cfg["model"]["fusion_dim"]
        self.text_enc   = TextEncoder(embed_dim=embed_dim)
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_classes(self, device):
        """Build class prototype embeddings from text descriptions."""
        class_embeds = []
        for cls_id in sorted(CLASS_DESCRIPTIONS.keys()):
            texts  = CLASS_DESCRIPTIONS[cls_id]
            embeds = self.text_enc(texts).to(device)
            proto  = embeds.mean(dim=0)            # average descriptions
            class_embeds.append(proto)
        return torch.stack(class_embeds, dim=0)    # [n_classes, embed_dim]

    def forward(self, image_feats):
        """
        image_feats : [B, embed_dim] CLS tokens
        returns     : [B, n_classes] logits
        """
        device      = image_feats.device
        img_proj    = F.normalize(self.image_proj(image_feats), dim=-1)
        cls_embeds  = self.encode_classes(device)       # [3, embed_dim]
        cls_embeds  = F.normalize(cls_embeds, dim=-1)

        # Cosine similarity scaled by temperature
        logits = (img_proj @ cls_embeds.T) / self.temperature.clamp(0.01, 1.0)
        return logits


# ─────────────────────────────────────────────
# 4. Zero-Shot Evaluator
# ─────────────────────────────────────────────
class ZeroShotEvaluator:
    def __init__(self, cfg, model):
        self.cfg       = cfg
        self.model     = model
        self.device    = torch.device(cfg["project"]["device"])
        self.zs_clf    = ZeroShotClassifier(cfg).to(self.device)

    @torch.no_grad()
    def evaluate(self, test_dataset):
        self.model.eval()
        self.zs_clf.eval()

        all_preds, all_labels = [], []

        for idx in range(len(test_dataset)):
            sample     = test_dataset[idx]
            image_seq  = sample["image_seq"].unsqueeze(0).to(self.device)
            clinical   = sample["clinical"].unsqueeze(0).to(self.device)
            valid_mask = sample["valid_mask"].unsqueeze(0).to(self.device)

            # Extract CLS token
            out     = self.model(image_seq, clinical, valid_mask)
            cls_tok = out["cls_token"]                  # [1, 256]

            # Zero-shot classify
            logits  = self.zs_clf(cls_tok)              # [1, 3]
            pred    = logits.argmax(dim=1).cpu().item()

            all_preds.append(pred)
            all_labels.append(sample["label"].item())

        # Compute accuracy
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        acc     = correct / max(len(all_labels), 1) * 100

        # Per-class accuracy
        per_class = {}
        class_names = {0: "Wildtype", 1: "Mutated", 2: "NOS/NEC"}
        for c in range(3):
            c_labels = [l for l in all_labels if l == c]
            c_preds  = [p for p, l in zip(all_preds, all_labels) if l == c]
            if c_labels:
                c_acc = sum(p == c for p in c_preds) / len(c_labels) * 100
                per_class[class_names[c]] = round(c_acc, 2)

        return {
            "zero_shot_accuracy": round(acc, 2),
            "per_class"         : per_class,
            "predictions"       : all_preds,
            "labels"            : all_labels,
        }


# ── Test ─────────────────────────────────────
if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["project"]["device"])

    print("="*50)
    print("  Phase 5 - Zero-Shot Evaluation")
    print("="*50)

    # Load trained model
    model = MedViTMT(cfg).to(device)
    ckpt  = torch.load(
        "outputs/checkpoints/best.pth",
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint (epoch {ckpt['epoch']},"
          f" val_loss={ckpt['val_loss']:.4f})")

    # Load test dataset
    test_ds = GBMDataset(cfg, split="test")

    # Run zero-shot evaluation
    evaluator = ZeroShotEvaluator(cfg, model)
    results   = evaluator.evaluate(test_ds)

    print(f"\nZero-Shot Accuracy : {results['zero_shot_accuracy']}%")
    print(f"Per-Class Accuracy :")
    for cls, acc in results["per_class"].items():
        print(f"  {cls}: {acc}%")