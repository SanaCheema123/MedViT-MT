"""
dataset.py
Phase 2 - PyTorch Dataset Class
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import yaml


# ─────────────────────────────────────────────
# 1. Augmentation pipelines
# ─────────────────────────────────────────────
def get_transforms(split="train", image_size=224):
    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:  # val / test — no augmentation
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 2. GBM Dataset class
# ─────────────────────────────────────────────
class GBMDataset(Dataset):
    """
    Returns per-patient:
      - image_seq   : Tensor [T, 3, H, W]  (T = num slices)
      - clinical    : Tensor [C]            (C = num clinical features)
      - slice_pos   : Tensor [T]            (temporal positions 0..T-1)
      - label       : int                   (IDH1 class for classification)
      - survival    : float                 (survival days, normalized)
      - patient_id  : str
    """

    CLINICAL_COLS = [
        "Gender",
        "Age_at_scan_years",
        "Survival_from_surgery_days_UPDATED",
        "IDH1",
        "MGMT",
        "GTR_over90percent",
        "PsP_TP_score",
    ]

    def __init__(self, cfg, split="train"):
        self.cfg        = cfg
        self.split      = split
        self.image_size = cfg["data"]["image_size"]
        self.max_slices = cfg["data"]["max_slices_per_patient"]
        self.transform  = get_transforms(split, self.image_size)

        # Load split IDs
        split_path = Path(cfg["paths"]["splits_dir"]) / f"{split}.json"
        with open(split_path, "r") as f:
            self.patient_ids = json.load(f)

        # Load processed clinical CSV
        clinical_path = Path(cfg["paths"]["processed_clinical"]) / "clinical_processed.csv"
        self.clinical_df = pd.read_csv(clinical_path)
        self.clinical_df.set_index("ID", inplace=True)

        # MRI processed dir
        self.mri_dir = Path(cfg["paths"]["processed_images"])

        print(f"[Dataset] {split.upper()} — {len(self.patient_ids)} patients loaded")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # ── Load MRI slice sequence ────────────
        patient_mri_dir = self.mri_dir / pid
        slices = sorted(patient_mri_dir.glob("*.png"))

        # Pad or truncate to max_slices
        slices = list(slices)
        if len(slices) > self.max_slices:
            start = (len(slices) - self.max_slices) // 2
            slices = slices[start: start + self.max_slices]

        images = []
        for sl in slices:
            img = Image.open(sl).convert("RGB")
            img = self.transform(img)
            images.append(img)

        # Pad sequence if fewer than max_slices
        T_actual = len(images)
        while len(images) < self.max_slices:
            images.append(torch.zeros(3, self.image_size, self.image_size))

        image_seq  = torch.stack(images, dim=0)         # [T, 3, H, W]
        slice_pos  = torch.arange(self.max_slices)       # [T]
        valid_mask = torch.zeros(self.max_slices)
        valid_mask[:T_actual] = 1.0                      # 1 = real, 0 = padding

        # ── Load clinical features ─────────────
        if pid in self.clinical_df.index:
            row = self.clinical_df.loc[pid]
            clinical_vals = []
            for col in self.CLINICAL_COLS:
                val = row[col] if col in row.index else 0.0
                clinical_vals.append(float(val) if pd.notna(val) else 0.0)
            clinical = torch.tensor(clinical_vals, dtype=torch.float32)

            label    = int(row["IDH1"]) if pd.notna(row.get("IDH1", 0)) else 0
            survival = float(row["Survival_from_surgery_days_UPDATED"]) if pd.notna(
                row.get("Survival_from_surgery_days_UPDATED", 0)) else 0.0
        else:
            clinical = torch.zeros(len(self.CLINICAL_COLS), dtype=torch.float32)
            label    = 0
            survival = 0.0

        return {
            "image_seq"  : image_seq,           # [T, 3, H, W]
            "clinical"   : clinical,             # [C]
            "slice_pos"  : slice_pos,            # [T]
            "valid_mask" : valid_mask,           # [T] padding mask
            "label"      : torch.tensor(label, dtype=torch.long),
            "survival"   : torch.tensor(survival, dtype=torch.float32),
            "patient_id" : pid,
        }


# ─────────────────────────────────────────────
# 3. DataLoader factory
# ─────────────────────────────────────────────
def get_dataloaders(cfg):
    batch_size = cfg["training"]["batch_size"]

    train_ds = GBMDataset(cfg, split="train")
    val_ds   = GBMDataset(cfg, split="val")
    test_ds  = GBMDataset(cfg, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # 0 for Windows compatibility
        pin_memory=False,       # keep False for CPU / low RAM
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"\n[DataLoader] Train batches : {len(train_loader)}")
    print(f"[DataLoader] Val   batches : {len(val_loader)}")
    print(f"[DataLoader] Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 4. Quick sanity check
# ─────────────────────────────────────────────
def verify_dataset(cfg):
    print("\n" + "="*50)
    print("  Dataset Sanity Check")
    print("="*50)

    for split in ["train", "val", "test"]:
        ds = GBMDataset(cfg, split=split)
        sample = ds[0]
        print(f"\n[{split.upper()}] First sample:")
        print(f"  patient_id  : {sample['patient_id']}")
        print(f"  image_seq   : {sample['image_seq'].shape}")
        print(f"  clinical    : {sample['clinical'].shape} → {sample['clinical']}")
        print(f"  slice_pos   : {sample['slice_pos'].shape}")
        print(f"  label       : {sample['label'].item()}")
        print(f"  survival    : {sample['survival'].item():.4f}")
        print(f"  valid_mask  : {sample['valid_mask'].sum().int().item()} real slices")


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Run sanity check after preprocessing is done
    verify_dataset(cfg)

    # Test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    batch = next(iter(train_loader))
    print(f"\n[Batch check] image_seq : {batch['image_seq'].shape}")
    print(f"[Batch check] clinical  : {batch['clinical'].shape}")
    print(f"[Batch check] label     : {batch['label']}")