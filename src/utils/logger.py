"""
logger.py
Phase 4 - Training Logger + Checkpoint Manager
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import os
import json
import logging
import torch
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 1. Logger setup
# ─────────────────────────────────────────────
def get_logger(cfg, name="MedViTMT"):
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"
    ))

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logger initialized -> {log_file}")
    return logger


# ─────────────────────────────────────────────
# 2. Metrics tracker
# ─────────────────────────────────────────────
class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_seg" : [], "val_seg"  : [],
            "train_cap" : [], "val_cap"  : [],
            "train_gnd" : [], "val_gnd"  : [],
            "train_cls" : [], "val_cls"  : [],
            "lr"        : [],
        }

    def update(self, split, losses, lr=None):
        self.history[f"{split}_loss"].append(losses.get("total", 0))
        self.history[f"{split}_seg" ].append(losses.get("seg",   0))
        self.history[f"{split}_cap" ].append(losses.get("cap",   0))
        self.history[f"{split}_gnd" ].append(losses.get("gnd",   0))
        self.history[f"{split}_cls" ].append(losses.get("cls",   0))
        if lr is not None:
            self.history["lr"].append(lr)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, path):
        with open(path) as f:
            self.history = json.load(f)


# ─────────────────────────────────────────────
# 3. Checkpoint manager
# ─────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, cfg, logger):
        self.ckpt_dir      = Path(cfg["paths"]["checkpoints"])
        self.save_best     = cfg["training"]["save_best_only"]
        self.logger        = logger
        self.best_val_loss = float("inf")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, val_loss, tracker):
        state = {
            "epoch"     : epoch,
            "val_loss"  : val_loss,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "scheduler" : scheduler.state_dict() if scheduler else None,
            "history"   : tracker.history,
        }

        # Always save latest
        latest_path = self.ckpt_dir / "latest.pth"
        torch.save(state, latest_path)

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.ckpt_dir / "best.pth"
            torch.save(state, best_path)
            self.logger.info(
                f"  New best model saved -> val_loss={val_loss:.4f}"
            )
            return True
        return False

    def load(self, model, optimizer=None, scheduler=None, best=True):
        path = self.ckpt_dir / ("best.pth" if best else "latest.pth")
        if not path.exists():
            self.logger.warning(f"No checkpoint found at {path}")
            return 0

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])

        self.best_val_loss = state.get("val_loss", float("inf"))
        self.logger.info(
            f"Checkpoint loaded from epoch {state['epoch']} "
            f"(val_loss={state['val_loss']:.4f})"
        )
        return state["epoch"]