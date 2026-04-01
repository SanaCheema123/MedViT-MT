"""
plots.py
Phase 6 - Individual Plots (one per metric)
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
Saves each plot as separate PNG file
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, auc


# ─────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "figure.facecolor" : "white",
        "axes.facecolor"   : "white",
        "axes.grid"        : True,
        "grid.alpha"       : 0.3,
        "font.size"        : 12,
        "axes.titlesize"   : 14,
        "axes.labelsize"   : 12,
        "lines.linewidth"  : 2,
    })


# ─────────────────────────────────────────────
# 1. Loss Curves (train vs val - each task)
# ─────────────────────────────────────────────
def plot_loss_curves(history, out_dir):
    out_dir = Path(out_dir) / "loss_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    epochs = range(1, len(history["train_loss"]) + 1)

    tasks = {
        "total": ("Total Loss",       "royalblue",   "coral"),
        "seg"  : ("Segmentation Loss","forestgreen", "lightcoral"),
        "cap"  : ("Captioning Loss",  "darkorange",  "gold"),
        "gnd"  : ("Grounding Loss",   "purple",      "violet"),
        "cls"  : ("Classification Loss","brown",     "sandybrown"),
    }

    for key, (title, tc, vc) in tasks.items():
        train_key = f"train_{key}"
        val_key   = f"val_{key}"
        if train_key not in history:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history[train_key], color=tc,
                label="Train", marker="o", markersize=4)
        ax.plot(epochs, history[val_key],   color=vc,
                label="Val",   marker="s", markersize=4, linestyle="--")
        ax.set_title(f"{title} - Train vs Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        path = out_dir / f"loss_{key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, out_dir, title="Confusion Matrix"):
    out_dir = Path(out_dir) / "confusion_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm_arr.max() / 2
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()

    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# 3. ROC Curves (one per class + combined)
# ─────────────────────────────────────────────
def plot_roc_curves(labels, probs, class_names, out_dir):
    out_dir = Path(out_dir) / "roc_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    labels = np.array(labels)
    probs  = np.array(probs)
    colors = ["royalblue", "forestgreen", "darkorange"]

    # Per-class ROC
    auc_scores = {}
    for c, (name, color) in enumerate(zip(class_names, colors)):
        if c >= probs.shape[1]:
            continue
        binary = (labels == c).astype(int)
        if binary.sum() == 0 or (1 - binary).sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(binary, probs[:, c])
        roc_auc     = auc(fpr, tpr)
        auc_scores[name] = round(roc_auc * 100, 2)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {name}")
        ax.legend(loc="lower right")
        path = out_dir / f"roc_{name.lower().replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # Combined ROC
    fig, ax = plt.subplots(figsize=(8, 7))
    for c, (name, color) in enumerate(zip(class_names, colors)):
        if c >= probs.shape[1]:
            continue
        binary = (labels == c).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, c])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Classes")
    ax.legend(loc="lower right")
    path = out_dir / "roc_all_classes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

    return auc_scores


# ─────────────────────────────────────────────
# 4. Dice Score per Slice
# ─────────────────────────────────────────────
def plot_dice_curves(dice_per_slice, out_dir):
    out_dir = Path(out_dir) / "dice_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    # Dice per slice
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(dice_per_slice)), dice_per_slice,
            color="forestgreen", marker=".", markersize=3)
    ax.axhline(y=np.mean(dice_per_slice), color="red",
               linestyle="--", label=f"Mean={np.mean(dice_per_slice):.3f}")
    ax.set_title("Dice Score per Patient")
    ax.set_xlabel("Patient Index")
    ax.set_ylabel("Dice Score")
    ax.legend()
    path = out_dir / "dice_per_patient.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

    # Dice distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dice_per_slice, bins=20, color="forestgreen",
            edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(dice_per_slice), color="red",
               linestyle="--", label=f"Mean={np.mean(dice_per_slice):.3f}")
    ax.set_title("Dice Score Distribution")
    ax.set_xlabel("Dice Score")
    ax.set_ylabel("Count")
    ax.legend()
    path = out_dir / "dice_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# 5. Metrics Bar Chart
# ─────────────────────────────────────────────
def plot_metrics_bar(metrics_dict, out_dir, title="Classification Metrics"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    keys   = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics_dict.get(k, 0) for k in keys]
    colors = ["royalblue", "forestgreen", "darkorange", "purple"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=12)

    path = out_dir / "metrics_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# 6. Kaplan-Meier Survival Curve
# ─────────────────────────────────────────────
def plot_kaplan_meier(km_data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.step(km_data["times"], km_data["survival"],
            where="post", color="royalblue", lw=2,
            label=f"Overall (n={km_data['n_total']})")
    ax.fill_between(km_data["times"], km_data["survival"],
                    step="post", alpha=0.15, color="royalblue")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.legend()
    path = out_dir / "kaplan_meier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# 7. Few-Shot vs Zero-Shot Bar
# ─────────────────────────────────────────────
def plot_shot_comparison(zero_acc, few_shot_results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    labels = ["Zero-Shot"] + list(few_shot_results.keys())
    values = [zero_acc] + list(few_shot_results.values())
    colors = ["gray"] + ["royalblue", "forestgreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors[:len(labels)],
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-Shot vs Few-Shot Accuracy")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold")

    path = out_dir / "shot_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    print("plots.py loaded OK")