"""
few_shot.py
Phase 5 - Few-Shot Learning Module
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
Prototypical Network approach: k=1, k=5, k=10
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
# 1. Prototypical Few-Shot Classifier
# ─────────────────────────────────────────────
class PrototypicalFewShot(nn.Module):
    """
    Prototypical Network for few-shot classification.
    Uses CLS token from MedViTMT as feature embedding.

    Steps:
      1. Compute class prototypes from k support samples
      2. Classify query samples by nearest prototype
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Small projection head
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

    def compute_prototypes(self, support_feats, support_labels, n_classes):
        """
        support_feats  : [K*N, embed_dim]
        support_labels : [K*N]
        n_classes      : int
        returns        : [n_classes, 64]
        """
        protos = []
        proj_feats = self.proj(support_feats)   # [K*N, 64]

        for c in range(n_classes):
            mask  = (support_labels == c)
            if mask.sum() == 0:
                protos.append(torch.zeros(64, device=support_feats.device))
            else:
                protos.append(proj_feats[mask].mean(dim=0))

        return torch.stack(protos, dim=0)        # [n_classes, 64]

    def forward(self, query_feats, prototypes):
        """
        query_feats : [Q, embed_dim]
        prototypes  : [n_classes, 64]
        returns     : [Q, n_classes] logits
        """
        q = self.proj(query_feats)               # [Q, 64]

        # Euclidean distance to prototypes
        dists = torch.cdist(
            q.unsqueeze(0),
            prototypes.unsqueeze(0)
        ).squeeze(0)                             # [Q, n_classes]

        return -dists                            # negative distance = logits


# ─────────────────────────────────────────────
# 2. Few-Shot Evaluator
# ─────────────────────────────────────────────
class FewShotEvaluator:
    def __init__(self, cfg, model):
        self.cfg       = cfg
        self.model     = model
        self.device    = torch.device(cfg["project"]["device"])
        self.k_shots   = cfg["few_shot"]["k_shots"]   # [1, 5, 10]
        self.n_classes = 3                             # IDH1: 0,1,2
        self.proto_net = PrototypicalFewShot(
            embed_dim=cfg["model"]["fusion_dim"]
        ).to(self.device)

    @torch.no_grad()
    def extract_features(self, dataset, indices):
        """Extract CLS token features for given patient indices."""
        feats, labels = [], []
        for idx in indices:
            sample     = dataset[idx]
            image_seq  = sample["image_seq"].unsqueeze(0).to(self.device)
            clinical   = sample["clinical"].unsqueeze(0).to(self.device)
            valid_mask = sample["valid_mask"].unsqueeze(0).to(self.device)

            out = self.model(image_seq, clinical, valid_mask)
            feats.append(out["cls_token"].squeeze(0).cpu())
            labels.append(sample["label"])

        return torch.stack(feats), torch.stack(labels)

    def evaluate(self, test_dataset, n_episodes=20):
        """
        Run few-shot evaluation for each k in k_shots.
        Returns accuracy per k.
        """
        self.model.eval()
        results = {}

        all_indices = list(range(len(test_dataset)))

        for k in self.k_shots:
            episode_accs = []

            for ep in range(n_episodes):
                # Sample support set: k samples per class
                support_idx, query_idx = [], []
                torch.manual_seed(ep)

                for c in range(self.n_classes):
                    # Find indices with label c
                    class_idx = [
                        i for i in all_indices
                        if test_dataset[i]["label"].item() == c
                    ]
                    if len(class_idx) < k + 1:
                        continue

                    perm = torch.randperm(len(class_idx)).tolist()
                    support_idx.extend([class_idx[p] for p in perm[:k]])
                    query_idx.extend([class_idx[p] for p in perm[k:k+3]])

                if not support_idx or not query_idx:
                    continue

                # Extract features
                sup_feats, sup_labels = self.extract_features(
                    test_dataset, support_idx
                )
                qry_feats, qry_labels = self.extract_features(
                    test_dataset, query_idx
                )

                sup_feats  = sup_feats.to(self.device)
                sup_labels = sup_labels.to(self.device)
                qry_feats  = qry_feats.to(self.device)

                # Compute prototypes
                protos  = self.proto_net.compute_prototypes(
                    sup_feats, sup_labels, self.n_classes
                )
                logits  = self.proto_net(qry_feats, protos)
                preds   = logits.argmax(dim=1).cpu()
                acc     = (preds == qry_labels).float().mean().item()
                episode_accs.append(acc)

            mean_acc = sum(episode_accs) / max(len(episode_accs), 1)
            results[f"{k}-shot"] = round(mean_acc * 100, 2)
            print(f"  {k}-shot accuracy: {mean_acc*100:.2f}%")

        return results


# ── Test ─────────────────────────────────────
if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["project"]["device"])

    print("="*50)
    print("  Phase 5 - Few-Shot Evaluation")
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

    # Run few-shot evaluation
    evaluator = FewShotEvaluator(cfg, model)
    results   = evaluator.evaluate(test_ds, n_episodes=10)

    print("\nFew-Shot Results:")
    for k, acc in results.items():
        print(f"  {k}: {acc}%")