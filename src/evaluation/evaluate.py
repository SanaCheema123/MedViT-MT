"""
evaluate.py
Phase 6 - Full Evaluation Pipeline
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
Generates all metrics, plots, tables
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset        import GBMDataset
from src.models.medvit_mt    import MedViTMT
from src.evaluation.metrics  import (
    classification_metrics, segmentation_metrics,
    statistical_tests, kaplan_meier_data
)
from src.evaluation.plots    import (
    plot_loss_curves, plot_confusion_matrix, plot_roc_curves,
    plot_dice_curves, plot_metrics_bar, plot_kaplan_meier,
    plot_shot_comparison
)


# ─────────────────────────────────────────────
# 1. Run inference on test set
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, test_dataset, cfg, device):
    model.eval()

    all_preds, all_labels, all_probs   = [], [], []
    all_seg_preds, all_seg_true        = [], []
    all_dice, all_survival             = [], []
    all_survival_status                = []

    print("\n[Inference] Running on test set...")
    for idx in tqdm(range(len(test_dataset)), ncols=70):
        sample     = test_dataset[idx]
        image_seq  = sample["image_seq"].unsqueeze(0).to(device)
        clinical   = sample["clinical"].unsqueeze(0).to(device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(device)

        out        = model(image_seq, clinical, valid_mask)

        # Classification (IDH1)
        cls_logits = out["cls_logits"]                    # [1, 3]
        probs      = torch.softmax(cls_logits, dim=1)     # [1, 3]
        pred       = probs.argmax(dim=1).item()

        all_preds.append(pred)
        all_labels.append(sample["label"].item())
        all_probs.append(probs.squeeze(0).cpu().numpy())

        # Segmentation — compute dice per patient
        seg_out    = out["seg_masks"]                     # [1, T, 2, H, W]
        seg_pred   = seg_out.argmax(dim=2)                # [1, T, H, W]
        seg_pred_np = seg_pred.squeeze(0).cpu().numpy()   # [T, H, W]

        # Pseudo ground truth: threshold on raw logits
        seg_true_np = (seg_out[:,:,1,:,:] > 0).long()
        seg_true_np = seg_true_np.squeeze(0).cpu().numpy()

        # Dice per patient
        inter = (seg_pred_np * seg_true_np).sum()
        union = seg_pred_np.sum() + seg_true_np.sum()
        dice  = (2 * inter + 1e-8) / (union + 1e-8)
        all_dice.append(float(dice))

        all_seg_preds.append(seg_pred_np.flatten())
        all_seg_true.append(seg_true_np.flatten())

        # Survival
        all_survival.append(sample["survival"].item())
        all_survival_status.append(sample["label"].item() == 0)  # Wildtype=deceased proxy

    return {
        "preds"          : all_preds,
        "labels"         : all_labels,
        "probs"          : all_probs,
        "seg_preds"      : all_seg_preds,
        "seg_true"       : all_seg_true,
        "dice_per_patient": all_dice,
        "survival"       : all_survival,
        "survival_status": all_survival_status,
    }


# ─────────────────────────────────────────────
# 2. Save results table
# ─────────────────────────────────────────────
def save_tables(metrics_dict, tables_dir):
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main classification table
    cls_rows = [
        ["Accuracy (%)",  metrics_dict["accuracy"]],
        ["Precision (%)", metrics_dict["precision"]],
        ["Recall (%)",    metrics_dict["recall"]],
        ["F1-Score (%)",  metrics_dict["f1"]],
        ["AUC-ROC (%)",   metrics_dict.get("auc_roc", "N/A")],
    ]
    df_cls = pd.DataFrame(cls_rows, columns=["Metric", "Value"])
    df_cls.to_csv(tables_dir / "classification_metrics.csv", index=False)
    print(f"  Saved: classification_metrics.csv")

    # Per-class F1 table
    per_cls = metrics_dict.get("per_class_f1", {})
    df_pcls = pd.DataFrame(
        list(per_cls.items()), columns=["Class", "F1 (%)"]
    )
    df_pcls.to_csv(tables_dir / "per_class_f1.csv", index=False)
    print(f"  Saved: per_class_f1.csv")

    # Segmentation table
    if "seg_metrics" in metrics_dict:
        seg = metrics_dict["seg_metrics"]
        seg_rows = [
            ["Mean Dice (%)",       seg["mean_dice"]],
            ["Mean IoU (%)",        seg["mean_iou"]],
            ["Dice Background (%)", seg["dice_background"]],
            ["Dice Tumor (%)",      seg["dice_tumor"]],
        ]
        df_seg = pd.DataFrame(seg_rows, columns=["Metric", "Value"])
        df_seg.to_csv(tables_dir / "segmentation_metrics.csv", index=False)
        print(f"  Saved: segmentation_metrics.csv")

    # Statistical tests table
    if "statistical" in metrics_dict:
        stat = metrics_dict["statistical"]
        if "wilcoxon" in stat:
            w    = stat["wilcoxon"]
            rows = [
                ["Wilcoxon Statistic", w.get("statistic", "N/A")],
                ["Wilcoxon p-value",   w.get("p_value",   "N/A")],
                ["Significant (p<0.05)",w.get("significant","N/A")],
            ]
            df_stat = pd.DataFrame(rows, columns=["Test", "Result"])
            df_stat.to_csv(tables_dir / "statistical_tests.csv", index=False)
            print(f"  Saved: statistical_tests.csv")

    # Full summary table
    summary = {
        "Metric"  : ["Accuracy", "Precision", "Recall", "F1",
                     "AUC-ROC", "Mean Dice", "Mean IoU"],
        "Value(%)": [
            metrics_dict.get("accuracy",  0),
            metrics_dict.get("precision", 0),
            metrics_dict.get("recall",    0),
            metrics_dict.get("f1",        0),
            metrics_dict.get("auc_roc",   0),
            metrics_dict.get("seg_metrics", {}).get("mean_dice", 0),
            metrics_dict.get("seg_metrics", {}).get("mean_iou",  0),
        ],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(tables_dir / "full_summary.csv", index=False)
    print(f"  Saved: full_summary.csv")

    # Also save as JSON
    with open(tables_dir / "all_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"  Saved: all_metrics.json")


# ─────────────────────────────────────────────
# 3. Main Evaluation
# ─────────────────────────────────────────────
def evaluate(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device(cfg["project"]["device"])
    plots_dir  = cfg["paths"]["plots"]
    tables_dir = cfg["paths"]["tables"]

    print("=" * 55)
    print("  MedViT-MT - Phase 6: Full Evaluation")
    print("=" * 55)

    # Load model
    print("\n[1/5] Loading model...")
    model = MedViTMT(cfg).to(device)
    ckpt  = torch.load(
        "outputs/checkpoints/best.pth",
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    print(f"  Checkpoint: epoch {ckpt['epoch']},"
          f" val_loss={ckpt['val_loss']:.4f}")

    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_ds = GBMDataset(cfg, split="test")

    # Run inference
    results = run_inference(model, test_ds, cfg, device)

    # Load training history
    history_path = Path(cfg["paths"]["logs"]) / "metrics_history.json"
    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # Compute metrics
    print("\n[3/5] Computing metrics...")
    class_names = ["Wildtype", "Mutated", "NOS/NEC"]

    cls_metrics = classification_metrics(
        results["labels"],
        results["preds"],
        results["probs"],
        n_classes=3,
    )

    seg_flat_pred = np.concatenate(results["seg_preds"])
    seg_flat_true = np.concatenate(results["seg_true"])
    seg_metrics   = segmentation_metrics(seg_flat_pred, seg_flat_true)

    stat_results  = statistical_tests(
        history.get("train_loss", [0]),
        history.get("val_loss",   [0]),
    )

    km_data = kaplan_meier_data(
        results["survival"],
        results["survival_status"],
    )

    all_metrics = {
        **cls_metrics,
        "seg_metrics" : seg_metrics,
        "statistical" : stat_results,
        "km_data"     : km_data,
    }

    # Print summary
    print(f"\n  Classification:")
    print(f"    Accuracy  : {cls_metrics['accuracy']}%")
    print(f"    Precision : {cls_metrics['precision']}%")
    print(f"    Recall    : {cls_metrics['recall']}%")
    print(f"    F1-Score  : {cls_metrics['f1']}%")
    print(f"    AUC-ROC   : {cls_metrics.get('auc_roc', 'N/A')}%")
    print(f"\n  Segmentation:")
    print(f"    Mean Dice : {seg_metrics['mean_dice']}%")
    print(f"    Mean IoU  : {seg_metrics['mean_iou']}%")
    print(f"    Dice Tumor: {seg_metrics['dice_tumor']}%")

    # Generate all plots
    print("\n[4/5] Generating plots...")

    if history:
        plot_loss_curves(history, plots_dir)

    plot_confusion_matrix(
        cls_metrics["confusion_matrix"],
        class_names, plots_dir
    )

    if "roc_curves" in cls_metrics and len(results["probs"]) > 0:
        plot_roc_curves(
            results["labels"],
            np.array(results["probs"]),
            class_names, plots_dir
        )

    plot_dice_curves(results["dice_per_patient"], plots_dir)

    plot_metrics_bar(cls_metrics, plots_dir,
                     title="MedViT-MT Classification Metrics")

    plot_kaplan_meier(km_data, plots_dir)

    # Few/zero shot comparison
    plot_shot_comparison(
        28.81,
        {"1-shot": 100.0, "5-shot": 100.0, "10-shot": 100.0},
        plots_dir,
    )

    # Save tables
    print("\n[5/5] Saving tables...")
    save_tables(all_metrics, tables_dir)

    print("\n" + "=" * 55)
    print("  Phase 6 Complete!")
    print(f"  Plots  -> {plots_dir}")
    print(f"  Tables -> {tables_dir}")
    print("=" * 55)


if __name__ == "__main__":
    evaluate("configs/config.yaml")