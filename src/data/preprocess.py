"""
preprocess.py
Phase 2 - Data Preprocessing Pipeline
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import os
import csv
import json
import shutil
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Load config
# ─────────────────────────────────────────────
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. Clinical data preprocessing
# ─────────────────────────────────────────────
def preprocess_clinical(cfg):
    csv_path = cfg["paths"]["clinical_csv"]
    out_dir   = Path(cfg["paths"]["processed_clinical"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    log.info(f"Loaded clinical CSV: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Replace 'Not Available' with NaN ──────
    df.replace("Not Available", np.nan, inplace=True)
    df.replace("Not Applicable", np.nan, inplace=True)

    # ── Encode categorical columns ─────────────
    encodings = {}

    # Gender: F=0, M=1
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1})
    encodings["Gender"] = {"F": 0, "M": 1}

    # IDH1: Wildtype=0, Mutated=1, NOS/NEC=2
    idh1_map = {"Wildtype": 0, "Mutated": 1, "NOS/NEC": 2}
    df["IDH1"] = df["IDH1"].map(idh1_map)
    encodings["IDH1"] = idh1_map

    # MGMT: Unmethylated=0, Methylated=1, Indeterminate=2
    mgmt_map = {"Unmethylated": 0, "Methylated": 1, "Indeterminate": 2}
    df["MGMT"] = df["MGMT"].map(mgmt_map)
    encodings["MGMT"] = mgmt_map

    # GTR_over90percent: Y=1, N=0
    df["GTR_over90percent"] = df["GTR_over90percent"].map({"Y": 1, "N": 0})
    encodings["GTR_over90percent"] = {"Y": 1, "N": 0}

    # Survival_Status: Deceased=0, Alive=1, Lost=2
    surv_map = {"Deceased": 0, "Alive": 1, "Lost to Follow-up": 2}
    df["Survival_Status"] = df["Survival_Status"].map(surv_map)
    encodings["Survival_Status"] = surv_map

    # ── Numeric columns: median imputation ────
    numeric_cols = [
        "Age_at_scan_years",
        "Survival_from_surgery_days_UPDATED",
        "GTR_over90percent",
        "Time_since_baseline_preop",
        "MGMT",
        "KPS",
        "IDH1",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median(skipna=True)
            missing = df[col].isna().sum()
            if missing > 0:
                df[col].fillna(median_val, inplace=True)
                log.info(f"  Imputed {missing} missing in '{col}' => median={median_val:.2f}")

    # PsP_TP_score: fill NA with 0
    if "PsP_TP_score" in df.columns:
        df["PsP_TP_score"] = pd.to_numeric(df["PsP_TP_score"], errors="coerce").fillna(0)

    # ── Normalize numeric features to [0,1] ──
    norm_cols = ["Age_at_scan_years", "Survival_from_surgery_days_UPDATED", "Time_since_baseline_preop"]
    norm_stats = {}
    for col in norm_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            df[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)
            norm_stats[col] = {"min": col_min, "max": col_max}

    # ── Save cleaned CSV ───────────────────────
    out_csv = out_dir / "clinical_processed.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Saved processed clinical CSV → {out_csv}")

    # Save encoding map for reference
    meta = {"encodings": encodings, "normalization": norm_stats}
    with open(out_dir / "clinical_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Clinical preprocessing complete.")
    return df


# ─────────────────────────────────────────────
# 3. MRI image preprocessing
# ─────────────────────────────────────────────
def preprocess_mri(cfg, patient_ids):
    mri_root   = Path(cfg["paths"]["mri_images"])
    subfolder  = cfg["paths"]["mri_subfolder"]          # images_DSC
    out_dir    = Path(cfg["paths"]["processed_images"])
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size  = cfg["data"]["image_size"]             # 224
    max_slices  = cfg["data"]["max_slices_per_patient"]  # 30

    patient_slice_map = {}   # { patient_id: [list of processed image paths] }
    missing_patients  = []

    for pid in patient_ids:
        dsc_folder = mri_root / pid if not subfolder else mri_root / pid / subfolder
        if not dsc_folder.exists():
            log.warning(f"  Missing DSC folder for {pid}: {dsc_folder}")
            missing_patients.append(pid)
            continue

        # Get sorted slice files
        slices = sorted(dsc_folder.glob("*.png"))
        if len(slices) == 0:
            slices = sorted(dsc_folder.glob("*.jpg"))
        if len(slices) == 0:
            log.warning(f"  No images found in {dsc_folder}")
            missing_patients.append(pid)
            continue

        # Cap to max_slices (take middle section — most informative)
        if len(slices) > max_slices:
            start = (len(slices) - max_slices) // 2
            slices = slices[start: start + max_slices]

        # Process each slice
        out_patient_dir = out_dir / pid
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        processed_paths = []
        for sl in slices:
            out_path = out_patient_dir / sl.name
            if out_path.exists():
                processed_paths.append(str(out_path))
                continue

            img = Image.open(sl).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)

            # Normalize pixel values to [0,1] and back to uint8
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)   # z-score per image
            arr = np.clip((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255, 0, 255)
            Image.fromarray(arr.astype(np.uint8)).save(out_path)
            processed_paths.append(str(out_path))

        patient_slice_map[pid] = processed_paths
        log.info(f"  {pid}: {len(processed_paths)} slices processed")

    log.info(f"MRI preprocessing complete. {len(patient_slice_map)} patients OK, {len(missing_patients)} missing.")
    if missing_patients:
        log.warning(f"  Missing: {missing_patients}")

    return patient_slice_map


# ─────────────────────────────────────────────
# 4. Train / Val / Test split
# ─────────────────────────────────────────────
def create_splits(cfg, df, patient_slice_map):
    splits_dir = Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    stratify_col = cfg["data"]["stratify_by"]   # IDH1
    seed         = cfg["project"]["seed"]
    train_r      = cfg["data"]["train_ratio"]
    val_r        = cfg["data"]["val_ratio"]

    # Only keep patients who have MRI images
    valid_ids = [pid for pid in df["ID"].tolist() if pid in patient_slice_map]
    log.info(f"Patients with both clinical + MRI: {len(valid_ids)}")

    # Stratified split by IDH1 class
    strat_groups = defaultdict(list)
    for pid in valid_ids:
        row = df[df["ID"] == pid]
        label = int(row[stratify_col].values[0]) if not row.empty else 0
        strat_groups[label].append(pid)

    train_ids, val_ids, test_ids = [], [], []
    random.seed(seed)

    for label, ids in strat_groups.items():
        random.shuffle(ids)
        n_train = max(1, int(len(ids) * train_r))
        n_val   = max(1, int(len(ids) * val_r))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train: n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split_name, ids in splits.items():
        out_path = splits_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(ids, f, indent=2)
        log.info(f"  {split_name}: {len(ids)} patients → {out_path}")

    return splits


# ─────────────────────────────────────────────
# 5. Main pipeline
# ─────────────────────────────────────────────
def run_preprocessing(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    log.info("=" * 55)
    log.info("  MedViT-MT — Phase 2: Preprocessing Pipeline")
    log.info("=" * 55)

    # Step A: Clinical
    log.info("\n[1/3] Preprocessing clinical data...")
    df = preprocess_clinical(cfg)

    # Step B: MRI
    log.info("\n[2/3] Preprocessing MRI images (images_DSC)...")
    patient_ids = df["ID"].tolist()
    patient_slice_map = preprocess_mri(cfg, patient_ids)

    # Step C: Splits
    log.info("\n[3/3] Creating train/val/test splits...")
    splits = create_splits(cfg, df, patient_slice_map)

    # Summary
    log.info("\n" + "=" * 55)
    log.info("  Preprocessing complete!")
    log.info(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    log.info("=" * 55)

    return df, patient_slice_map, splits


if __name__ == "__main__":
    run_preprocessing("configs/config.yaml")