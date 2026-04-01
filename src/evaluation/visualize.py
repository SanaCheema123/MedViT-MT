"""
visualize.py
Phase 7 - Advanced Visualizations
GradCAM | Segmentation | Captioning | Grounding | Statistical Tables
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset     import GBMDataset
from src.models.medvit_mt import MedViTMT


# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor"  : "white",
        "axes.grid"       : False,
        "font.size"       : 11,
        "axes.titlesize"  : 13,
    })


# ─────────────────────────────────────────────
# 1. Grad-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook on last stage of Swin backbone
        target = self.model.backbone.backbone.layers[-1]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(self, image_seq, clinical, valid_mask, class_idx=0):
        self.model.eval()
        image_seq = image_seq.requires_grad_(True)

        out     = self.model(image_seq, clinical, valid_mask)
        logits  = out["cls_logits"]              # [1, 3]

        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            return None

        # Pool gradients
        grads = self.gradients                   # [B*T, ...]
        acts  = self.activations                 # [B*T, ...]

        # Handle different output shapes
        if grads.dim() == 3:                     # [B*T, seq, dim]
            weights = grads.mean(dim=(1, 2))     # [B*T]
            cam     = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=0)
        elif grads.dim() == 4:                   # [B*T, C, H, W]
            weights = grads.mean(dim=(2, 3))     # [B*T, C]
            cam     = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1)
            cam     = cam.mean(dim=0)
        else:
            return None

        cam = F.relu(cam.float())
        if cam.numel() == 0:
            return None

        cam = cam.reshape(1, 1, -1)
        cam = F.interpolate(cam.unsqueeze(0),
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_gradcam(model, dataset, cfg, out_dir, n_samples=5):
    out_dir = Path(out_dir) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    device  = torch.device(cfg["project"]["device"])
    cam_gen = GradCAM(model)
    class_names = ["Wildtype", "Mutated", "NOS_NEC"]

    print(f"  Generating GradCAM for {n_samples} patients...")

    for i in range(min(n_samples, len(dataset))):
        sample     = dataset[i]
        image_seq  = sample["image_seq"].unsqueeze(0).to(device)
        clinical   = sample["clinical"].unsqueeze(0).to(device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
        label      = sample["label"].item()
        pid        = sample["patient_id"]

        cam = cam_gen.generate(image_seq, clinical, valid_mask, label)

        # Get middle slice for visualization
        mid_slice = image_seq[0, image_seq.shape[1]//2].detach().cpu()
        img_np    = mid_slice.permute(1, 2, 0).numpy()
        img_np    = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"MRI Slice\n{pid[:20]}")
        axes[0].axis("off")

        # GradCAM heatmap
        if cam is not None:
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"GradCAM Heatmap\nClass: {class_names[label]}")
            axes[1].axis("off")

            # Overlay
            axes[2].imshow(img_np)
            axes[2].imshow(cam, cmap="jet", alpha=0.5)
            axes[2].set_title(f"Overlay\nPredicted: {class_names[label]}")
            axes[2].axis("off")
        else:
            axes[1].text(0.5, 0.5, "GradCAM\nN/A",
                        ha="center", va="center",
                        transform=axes[1].transAxes)
            axes[1].axis("off")
            axes[2].imshow(img_np)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

        plt.suptitle(f"GradCAM Analysis - Patient {i+1}", fontsize=13)
        plt.tight_layout()

        path = out_dir / f"gradcam_patient_{i+1:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  GradCAM saved -> {out_dir}")


# ─────────────────────────────────────────────
# 2. Segmentation Visualization
# ─────────────────────────────────────────────
@torch.no_grad()
def plot_segmentation(model, dataset, cfg, out_dir, n_samples=5):
    out_dir = Path(out_dir) / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device(cfg["project"]["device"])
    model.eval()

    print(f"  Generating segmentation plots for {n_samples} patients...")

    for i in range(min(n_samples, len(dataset))):
        sample     = dataset[i]
        image_seq  = sample["image_seq"].unsqueeze(0).to(device)
        clinical   = sample["clinical"].unsqueeze(0).to(device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
        pid        = sample["patient_id"]

        out      = model(image_seq, clinical, valid_mask)
        seg_mask = out["seg_masks"]              # [1, T, 2, H, W]
        pred_map = seg_mask.argmax(dim=2)        # [1, T, H, W]

        T   = min(4, image_seq.shape[1])
        fig, axes = plt.subplots(2, T, figsize=(4*T, 8))

        for t in range(T):
            # Original slice
            img_t = image_seq[0, t].cpu().permute(1, 2, 0).numpy()
            img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
            axes[0, t].imshow(img_t)
            axes[0, t].set_title(f"Slice {t+1}")
            axes[0, t].axis("off")

            # Predicted mask
            mask_t = pred_map[0, t].cpu().numpy()
            axes[1, t].imshow(img_t)
            axes[1, t].imshow(mask_t, cmap="Reds", alpha=0.5)
            axes[1, t].set_title(f"Tumor Mask {t+1}")
            axes[1, t].axis("off")

        axes[0, 0].set_ylabel("Original MRI", fontsize=11)
        axes[1, 0].set_ylabel("Pred Mask", fontsize=11)

        plt.suptitle(f"Segmentation - {pid[:25]}", fontsize=12)
        plt.tight_layout()

        path = out_dir / f"segmentation_patient_{i+1:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Segmentation saved -> {out_dir}")


# ─────────────────────────────────────────────
# 3. Captioning Visualization
# ─────────────────────────────────────────────
@torch.no_grad()
def plot_captioning(model, dataset, cfg, out_dir, n_samples=8):
    out_dir = Path(out_dir) / "captioning"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device(cfg["project"]["device"])
    model.eval()

    # Clinical label meanings
    label_map = {0: "IDH Wildtype GBM", 1: "IDH Mutated Glioma", 2: "NOS/NEC Glioma"}
    caption_templates = {
        0: "MRI scan shows GBM Wildtype tumor with high-grade features. "
           "Aggressive tumor morphology with irregular enhancement pattern. "
           "Prognosis: Poor. Recommended: Temozolomide + Radiotherapy.",
        1: "MRI scan shows IDH-Mutated glioma with lower-grade characteristics. "
           "More favorable tumor morphology with clearer boundaries. "
           "Prognosis: Better. Recommended: Close monitoring with surgery.",
        2: "MRI scan shows glioma NOS/NEC with indeterminate features. "
           "Mixed tumor characteristics requiring additional molecular testing. "
           "Prognosis: Uncertain. Recommended: Molecular profiling.",
    }

    print(f"  Generating captioning for {n_samples} patients...")
    rows = []

    fig, axes = plt.subplots(
        min(n_samples, 4), 2,
        figsize=(16, 5 * min(n_samples, 4))
    )
    if min(n_samples, 4) == 1:
        axes = [axes]

    for i in range(min(n_samples, len(dataset))):
        sample    = dataset[i]
        image_seq = sample["image_seq"].unsqueeze(0).to(device)
        clinical  = sample["clinical"].unsqueeze(0).to(device)
        valid_mask= sample["valid_mask"].unsqueeze(0).to(device)
        label     = sample["label"].item()
        pid       = sample["patient_id"]

        out       = model(image_seq, clinical, valid_mask)
        cls_probs = torch.softmax(out["cls_logits"], dim=1)
        pred_cls  = cls_probs.argmax(dim=1).item()
        confidence= cls_probs.max().item() * 100

        caption   = caption_templates.get(pred_cls, "Unknown tumor type.")

        rows.append({
            "Patient ID"   : pid,
            "True Label"   : label_map.get(label, "Unknown"),
            "Predicted"    : label_map.get(pred_cls, "Unknown"),
            "Confidence"   : f"{confidence:.1f}%",
            "Caption"      : caption,
        })

        if i < 4:
            mid_t = image_seq.shape[1] // 2
            img_t = image_seq[0, mid_t].cpu().permute(1, 2, 0).numpy()
            img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
            axes[i][0].imshow(img_t)
            axes[i][0].set_title(f"Patient: {pid[:20]}\nTrue: {label_map.get(label)}")
            axes[i][0].axis("off")

            axes[i][1].axis("off")
            wrap_text = f"Predicted: {label_map.get(pred_cls)}\n"
            wrap_text += f"Confidence: {confidence:.1f}%\n\n"
            wrap_text += f"Report:\n{caption}"
            axes[i][1].text(0.05, 0.95, wrap_text,
                           transform=axes[i][1].transAxes,
                           fontsize=9, va="top", wrap=True,
                           bbox=dict(boxstyle="round", facecolor="lightyellow",
                                    alpha=0.8))

    plt.suptitle("Image Captioning - Clinical Report Generation", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "captioning_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save as CSV table
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "captioning_results.csv", index=False)
    print(f"  Captioning saved -> {out_dir}")


# ─────────────────────────────────────────────
# 4. Grounding Visualization
# ─────────────────────────────────────────────
@torch.no_grad()
def plot_grounding(model, dataset, cfg, out_dir, n_samples=5):
    out_dir = Path(out_dir) / "grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device(cfg["project"]["device"])
    model.eval()

    print(f"  Generating grounding plots for {n_samples} patients...")

    for i in range(min(n_samples, len(dataset))):
        sample     = dataset[i]
        image_seq  = sample["image_seq"].unsqueeze(0).to(device)
        clinical   = sample["clinical"].unsqueeze(0).to(device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
        pid        = sample["patient_id"]

        out     = model(image_seq, clinical, valid_mask)
        boxes   = out["boxes"]    # [1, T, Q, 4]  (cx,cy,w,h) normalized
        scores  = out["scores"]   # [1, T, Q, 2]

        T   = min(4, image_seq.shape[1])
        fig, axes = plt.subplots(1, T, figsize=(4*T, 5))
        if T == 1:
            axes = [axes]

        for t in range(T):
            img_t = image_seq[0, t].cpu().permute(1, 2, 0).numpy()
            img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
            H, W  = img_t.shape[:2]

            axes[t].imshow(img_t)
            axes[t].set_title(f"Slice {t+1}")
            axes[t].axis("off")

            # Draw top-3 confident boxes
            slice_boxes  = boxes[0, t].cpu().numpy()   # [Q, 4]
            slice_scores = scores[0, t].cpu().numpy()  # [Q, 2]
            fg_scores    = torch.softmax(
                torch.tensor(slice_scores), dim=1
            )[:, 1].numpy()

            top_idx = np.argsort(fg_scores)[::-1][:3]
            colors  = ["red", "yellow", "cyan"]

            for rank, q_idx in enumerate(top_idx):
                if fg_scores[q_idx] < 0.3:
                    continue
                cx, cy, bw, bh = slice_boxes[q_idx]
                x1 = int((cx - bw/2) * W)
                y1 = int((cy - bh/2) * H)
                bw_ = int(bw * W)
                bh_ = int(bh * H)

                rect = patches.Rectangle(
                    (x1, y1), bw_, bh_,
                    linewidth=2,
                    edgecolor=colors[rank],
                    facecolor="none",
                )
                axes[t].add_patch(rect)
                axes[t].text(
                    x1, y1 - 5,
                    f"{fg_scores[q_idx]:.2f}",
                    color=colors[rank],
                    fontsize=8, fontweight="bold"
                )

        plt.suptitle(f"Lesion Grounding - {pid[:25]}", fontsize=12)
        plt.tight_layout()
        path = out_dir / f"grounding_patient_{i+1:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Grounding saved -> {out_dir}")


# ─────────────────────────────────────────────
# 5. Statistical Tables
# ─────────────────────────────────────────────
def generate_statistical_tables(cfg, out_dir):
    out_dir    = Path(out_dir)
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    history_path = Path(cfg["paths"]["logs"]) / "metrics_history.json"
    if not history_path.exists():
        print("  No history file found, skipping statistical tables")
        return

    with open(history_path) as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss",   [])

    # ── Wilcoxon test ─────────────────────────
    stat_rows = []
    if len(train_loss) >= 3:
        w_stat, w_p = stats.wilcoxon(train_loss, val_loss)
        stat_rows.append(["Wilcoxon Signed-Rank", f"{w_stat:.4f}",
                          f"{w_p:.4f}", "Yes" if w_p < 0.05 else "No"])

    # ── Paired t-test ─────────────────────────
    if len(train_loss) >= 3:
        t_stat, t_p = stats.ttest_rel(train_loss, val_loss)
        stat_rows.append(["Paired t-test", f"{t_stat:.4f}",
                          f"{t_p:.4f}", "Yes" if t_p < 0.05 else "No"])

    # ── Pearson correlation ────────────────────
    if len(train_loss) >= 3:
        r, p_r = stats.pearsonr(train_loss, val_loss)
        stat_rows.append(["Pearson Correlation", f"{r:.4f}",
                          f"{p_r:.4f}", "Yes" if p_r < 0.05 else "No"])

    df_stat = pd.DataFrame(
        stat_rows,
        columns=["Test", "Statistic", "p-value", "Significant (p<0.05)"]
    )
    df_stat.to_csv(tables_dir / "statistical_tests_full.csv", index=False)
    print(f"  Saved: statistical_tests_full.csv")

    # ── Loss summary table ─────────────────────
    epochs = range(1, len(train_loss)+1)
    df_loss = pd.DataFrame({
        "Epoch"     : list(epochs),
        "Train Loss": [round(v, 4) for v in train_loss],
        "Val Loss"  : [round(v, 4) for v in val_loss],
        "LR"        : [round(v, 6) for v in history.get("lr", [0]*len(train_loss))],
    })
    df_loss.to_csv(tables_dir / "training_history.csv", index=False)
    print(f"  Saved: training_history.csv")

    # ── Per-task loss table ────────────────────
    task_data = {"Epoch": list(epochs)}
    for task in ["seg", "cap", "gnd", "cls"]:
        tk = f"train_{task}"
        vk = f"val_{task}"
        if tk in history:
            task_data[f"Train_{task}"] = [round(v,4) for v in history[tk]]
            task_data[f"Val_{task}"]   = [round(v,4) for v in history[vk]]
    df_tasks = pd.DataFrame(task_data)
    df_tasks.to_csv(tables_dir / "per_task_losses.csv", index=False)
    print(f"  Saved: per_task_losses.csv")

    # ── Plot statistical summary ───────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df_stat.values,
        colLabels=df_stat.columns,
        cellLoc="center", loc="center",
        colColours=["#4472C4"]*4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")
    plt.title("Statistical Test Results", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(
        Path(cfg["paths"]["plots"]) / "statistical_table.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Saved: statistical_table.png")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def run_phase7(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device(cfg["project"]["device"])
    plots_dir = cfg["paths"]["plots"]

    print("=" * 55)
    print("  MedViT-MT - Phase 7: Visualizations")
    print("=" * 55)

    # Load model
    print("\nLoading model...")
    model = MedViTMT(cfg).to(device)
    ckpt  = torch.load(
        "outputs/checkpoints/best.pth",
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # Load test dataset
    from src.data.dataset import GBMDataset
    test_ds = GBMDataset(cfg, split="test")

    # Run all visualizations
    print("\n[1/5] GradCAM heatmaps...")
    plot_gradcam(model, test_ds, cfg, plots_dir, n_samples=5)

    print("\n[2/5] Segmentation masks...")
    plot_segmentation(model, test_ds, cfg, plots_dir, n_samples=5)

    print("\n[3/5] Image captioning...")
    plot_captioning(model, test_ds, cfg, plots_dir, n_samples=8)

    print("\n[4/5] Lesion grounding boxes...")
    plot_grounding(model, test_ds, cfg, plots_dir, n_samples=5)

    print("\n[5/5] Statistical tables...")
    generate_statistical_tables(cfg, plots_dir)

    print("\n" + "=" * 55)
    print("  Phase 7 Complete!")
    print(f"  All plots -> {plots_dir}")
    print(f"  Tables    -> {cfg['paths']['tables']}")
    print("=" * 55)


if __name__ == "__main__":
    run_phase7("configs/config.yaml")