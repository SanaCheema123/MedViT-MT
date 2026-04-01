"""
train.py
Phase 4 - Main Training Script
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset      import get_dataloaders
from src.models.medvit_mt  import MedViTMT
from src.utils.losses      import MultiTaskLoss
from src.utils.logger      import get_logger, MetricsTracker, CheckpointManager


# ─────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# 2. Build dummy targets (no ground truth masks)
# ─────────────────────────────────────────────
def build_targets(batch, cfg, device):
    """
    Since we have no manual segmentation masks,
    we create pseudo targets from model outputs
    for initial training (self-supervised warm-up).
    Real masks can replace these when available.
    """
    B      = batch["image_seq"].shape[0]
    T      = cfg["data"]["max_slices_per_patient"]
    H = W  = cfg["data"]["image_size"]

    targets = {
        # Pseudo segmentation: all background (class 0)
        "seg_masks"  : torch.zeros(B, T, H, W, dtype=torch.long).to(device),
        # Pseudo caption tokens: all zeros (padding)
        "cap_tokens" : torch.zeros(B, 64, dtype=torch.long).to(device),
        # Labels from clinical IDH1
        "labels"     : batch["label"].to(device),
    }
    return targets


# ─────────────────────────────────────────────
# 3. Train one epoch
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn, cfg, device, logger):
    model.train()
    total_losses = {"total": 0, "seg": 0, "cap": 0, "gnd": 0, "cls": 0}
    n_batches    = 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=80)

    for batch in pbar:
        image_seq   = batch["image_seq"].to(device)    # [B, T, 3, H, W]
        clinical    = batch["clinical"].to(device)      # [B, 7]
        valid_mask  = batch["valid_mask"].to(device)    # [B, T]
        targets     = build_targets(batch, cfg, device)

        optimizer.zero_grad()

        outputs = model(image_seq, clinical, valid_mask)
        losses  = loss_fn(outputs, targets)

        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["training"]["grad_clip"]
        )

        optimizer.step()

        # Accumulate losses
        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    # Average losses
    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg


# ─────────────────────────────────────────────
# 4. Validation one epoch
# ─────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, loss_fn, cfg, device):
    model.eval()
    total_losses = {"total": 0, "seg": 0, "cap": 0, "gnd": 0, "cls": 0}
    n_batches    = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=80)

    for batch in pbar:
        image_seq  = batch["image_seq"].to(device)
        clinical   = batch["clinical"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        targets    = build_targets(batch, cfg, device)

        outputs = model(image_seq, clinical, valid_mask)
        losses  = loss_fn(outputs, targets)

        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg


# ─────────────────────────────────────────────
# 5. Main training loop
# ─────────────────────────────────────────────
def train(config_path="configs/config.yaml"):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"])

    # Logger
    logger = get_logger(cfg)
    logger.info("=" * 55)
    logger.info("  MedViT-MT — Phase 4: Training")
    logger.info("=" * 55)
    logger.info(f"  Device  : {device}")
    logger.info(f"  Epochs  : {cfg['training']['epochs']}")
    logger.info(f"  Batch   : {cfg['training']['batch_size']}")

    # Data
    logger.info("\nLoading datasets...")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model
    logger.info("Building model...")
    model = MedViTMT(cfg).to(device)
    total, trainable = model.count_parameters()
    logger.info(f"  Params: {total:,} total | {trainable:,} trainable")

    # Loss
    loss_fn = MultiTaskLoss(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )

    # Tracker + Checkpoint
    tracker = MetricsTracker()
    ckpt_mgr = CheckpointManager(cfg, logger)

    # ── Training loop ─────────────────────────
    logger.info("\nStarting training...\n")
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch:03d}/{epochs}] lr={lr:.6f}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, cfg, device, logger
        )
        logger.info(
            f"  Train -> total={train_losses['total']:.4f} | "
            f"seg={train_losses['seg']:.4f} | "
            f"cap={train_losses['cap']:.4f} | "
            f"gnd={train_losses['gnd']:.4f} | "
            f"cls={train_losses['cls']:.4f}"
        )

        # Validate
        val_losses = val_epoch(model, val_loader, loss_fn, cfg, device)
        logger.info(
            f"  Val   -> total={val_losses['total']:.4f} | "
            f"seg={val_losses['seg']:.4f} | "
            f"cap={val_losses['cap']:.4f} | "
            f"gnd={val_losses['gnd']:.4f} | "
            f"cls={val_losses['cls']:.4f}"
        )

        # Update tracker
        tracker.update("train", train_losses, lr)
        tracker.update("val",   val_losses)

        # Save checkpoint
        ckpt_mgr.save(model, optimizer, scheduler,
                      epoch, val_losses["total"], tracker)

        # Step scheduler
        scheduler.step()

        logger.info("")

    logger.info("=" * 55)
    logger.info("  Training complete!")
    logger.info(f"  Best val loss: {ckpt_mgr.best_val_loss:.4f}")
    logger.info(f"  Checkpoints -> {cfg['paths']['checkpoints']}")
    logger.info("=" * 55)

    # Save final metrics history
    tracker.save(Path(cfg["paths"]["logs"]) / "metrics_history.json")


if __name__ == "__main__":
    train("configs/config.yaml")