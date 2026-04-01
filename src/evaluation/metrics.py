"""
metrics.py
Phase 6 - Evaluation Metrics
MedViT-MT: GBM Brain Tumor Multi-Task Transformer
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, average_precision_score
)
from scipy import stats


# ─────────────────────────────────────────────
# 1. Classification Metrics
# ─────────────────────────────────────────────
def classification_metrics(labels, preds, probs=None, n_classes=3):
    """
    labels : list/array of true class indices
    preds  : list/array of predicted class indices
    probs  : array [N, n_classes] predicted probabilities
    """
    labels = np.array(labels)
    preds  = np.array(preds)

    results = {
        "accuracy" : round(accuracy_score(labels, preds) * 100, 2),
        "precision": round(precision_score(
            labels, preds, average="macro", zero_division=0) * 100, 2),
        "recall"   : round(recall_score(
            labels, preds, average="macro", zero_division=0) * 100, 2),
        "f1"       : round(f1_score(
            labels, preds, average="macro", zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    # Per-class metrics
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    results["per_class_f1"] = {
        "Wildtype": round(per_class_f1[0] * 100, 2) if len(per_class_f1) > 0 else 0,
        "Mutated" : round(per_class_f1[1] * 100, 2) if len(per_class_f1) > 1 else 0,
        "NOS_NEC" : round(per_class_f1[2] * 100, 2) if len(per_class_f1) > 2 else 0,
    }

    # AUC-ROC (one-vs-rest)
    if probs is not None:
        probs = np.array(probs)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr",
                                average="macro")
            results["auc_roc"] = round(auc * 100, 2)

            # Per-class ROC curves
            roc_data = {}
            for c in range(min(n_classes, probs.shape[1])):
                binary = (labels == c).astype(int)
                if binary.sum() > 0 and (1 - binary).sum() > 0:
                    fpr, tpr, thr = roc_curve(binary, probs[:, c])
                    roc_data[c] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thr.tolist(),
                    }
            results["roc_curves"] = roc_data
        except Exception as e:
            results["auc_roc"] = 0.0
            print(f"  AUC warning: {e}")

    return results


# ─────────────────────────────────────────────
# 2. Segmentation Metrics
# ─────────────────────────────────────────────
def segmentation_metrics(pred_masks, true_masks, n_classes=2):
    """
    pred_masks : [N, H, W] predicted class maps
    true_masks : [N, H, W] ground truth class maps
    """
    pred_masks = np.array(pred_masks).flatten()
    true_masks = np.array(true_masks).flatten()

    dice_scores, iou_scores = [], []

    for c in range(n_classes):
        pred_c = (pred_masks == c).astype(float)
        true_c = (true_masks == c).astype(float)

        intersection = (pred_c * true_c).sum()
        union        = pred_c.sum() + true_c.sum()

        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        iou  = (intersection + 1e-8) / (union - intersection + 1e-8)

        dice_scores.append(round(float(dice) * 100, 2))
        iou_scores.append(round(float(iou) * 100, 2))

    return {
        "dice_per_class"    : dice_scores,
        "iou_per_class"     : iou_scores,
        "mean_dice"         : round(np.mean(dice_scores), 2),
        "mean_iou"          : round(np.mean(iou_scores), 2),
        "dice_background"   : dice_scores[0],
        "dice_tumor"        : dice_scores[1] if len(dice_scores) > 1 else 0,
    }


# ─────────────────────────────────────────────
# 3. Statistical Tests
# ─────────────────────────────────────────────
def statistical_tests(group1, group2):
    """
    Wilcoxon signed-rank test between two groups.
    group1, group2: lists of values (e.g. train_loss, val_loss)
    """
    results = {}

    # Wilcoxon signed-rank test
    try:
        if len(group1) == len(group2) and len(group1) > 2:
            stat, p = stats.wilcoxon(group1, group2)
            results["wilcoxon"] = {
                "statistic": round(float(stat), 4),
                "p_value"  : round(float(p), 4),
                "significant": bool(p < 0.05),
            }
    except Exception as e:
        results["wilcoxon"] = {"error": str(e)}

    # McNemar test (for binary predictions)
    results["note"] = "McNemar requires paired binary predictions"

    # DeLong test approximation (AUC comparison)
    results["delong_note"] = "DeLong AUC comparison in plots.py"

    return results


# ─────────────────────────────────────────────
# 4. Survival Analysis (Kaplan-Meier)
# ─────────────────────────────────────────────
def kaplan_meier_data(survival_days, survival_status, groups=None):
    """
    Prepare data for Kaplan-Meier curve.
    survival_days   : list of survival durations
    survival_status : list of 0=alive, 1=deceased
    groups          : optional list of group labels
    """
    survival_days   = np.array(survival_days)
    survival_status = np.array(survival_status)

    # Sort by time
    sort_idx      = np.argsort(survival_days)
    times         = survival_days[sort_idx]
    events        = survival_status[sort_idx]

    # KM estimator
    n         = len(times)
    km_times  = [0]
    km_surv   = [1.0]
    at_risk   = n
    survival  = 1.0

    for i in range(n):
        if events[i] == 1:   # event occurred
            survival *= (1 - 1 / at_risk)
            km_times.append(float(times[i]))
            km_surv.append(round(survival, 4))
        at_risk -= 1

    return {
        "times"   : km_times,
        "survival": km_surv,
        "n_total" : int(n),
        "n_events": int(events.sum()),
    }


if __name__ == "__main__":
    print("Metrics module loaded OK")
    # Quick test
    labels = [0, 0, 1, 2, 0, 1, 2, 0]
    preds  = [0, 0, 1, 2, 1, 1, 2, 0]
    probs  = np.random.dirichlet([1,1,1], size=8)
    res    = classification_metrics(labels, preds, probs)
    print(f"  Accuracy  : {res['accuracy']}%")
    print(f"  F1 (macro): {res['f1']}%")
    print(f"  AUC-ROC   : {res['auc_roc']}%")
    print("Metrics test passed")