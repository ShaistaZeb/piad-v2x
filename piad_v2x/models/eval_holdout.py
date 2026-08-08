"""Held-out attack evaluation utilities.

Binary detection metrics (benign vs misbehaviour) for held-out attack types,
plus matplotlib plots of confusion matrices and per-class F1 bars.

The hypothesis under test is that the multi-physics PINN (L_kin + L_lwr)
recognises coordinated attack patterns that the kinematic-only baseline
misses. This module provides the comparison apparatus; the actual claim
is tested in `experiments/run_w4_holdout.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryReport:
    """Binary (benign vs misbehaviour) detection metrics."""
    accuracy: float
    precision_attack: float
    recall_attack: float
    f1_attack: float
    auc_roc: float
    n_attack: int
    n_benign: int

    def short(self) -> str:
        return (
            f"acc={self.accuracy:.4f}  "
            f"P_attack={self.precision_attack:.4f}  "
            f"R_attack={self.recall_attack:.4f}  "
            f"F1_attack={self.f1_attack:.4f}  "
            f"AUC={self.auc_roc:.4f}"
        )


def to_binary(y: np.ndarray) -> np.ndarray:
    """Collapse multi-class labels to {0=benign, 1=misbehaviour}."""
    return (y != 0).astype(np.int32)


def binary_report(
    y_true: np.ndarray,
    y_pred_class: np.ndarray,
    p_attack: np.ndarray | None = None,
) -> BinaryReport:
    """Compute binary detection metrics from multi-class predictions.

    `p_attack` is the predicted probability of misbehaviour (1 - p_benign)
    used to compute AUC. If absent, AUC is reported as NaN.
    """
    y_b = to_binary(y_true)
    pred_b = to_binary(y_pred_class)
    acc = float((pred_b == y_b).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_b, pred_b, average="binary", pos_label=1, zero_division=0
    )
    if p_attack is not None and len(np.unique(y_b)) > 1:
        try:
            auc = float(roc_auc_score(y_b, p_attack))
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")
    return BinaryReport(
        accuracy=acc, precision_attack=float(p), recall_attack=float(r),
        f1_attack=float(f1), auc_roc=auc,
        n_attack=int(y_b.sum()), n_benign=int((y_b == 0).sum()),
    )


@dataclass(frozen=True)
class PerClassReport:
    """Per-class metrics on the held-out attack classes only."""
    classes: tuple[int, ...]
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]
    support: tuple[int, ...]

    def table(self) -> str:
        rows = ["class  precision  recall  f1     support"]
        for c, p, r, f, s in zip(
            self.classes, self.precision, self.recall, self.f1, self.support
        ):
            rows.append(f"{c:>5}  {p:>9.3f}  {r:>6.3f}  {f:>5.3f}  {s:>7}")
        return "\n".join(rows)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: tuple[int, ...],
) -> PerClassReport:
    """Per-attack-class precision/recall/F1.

    For held-out attacks the model never saw, "correct" means it predicted
    SOME misbehaviour class (any non-zero label). Precision is computed
    over all predictions of that class; recall is the fraction of true
    held-out attacks that the model labelled non-zero.
    """
    precisions, recalls, f1s, supports = [], [], [], []
    for c in classes:
        true_mask = (y_true == c)
        pred_mask = (y_pred == c)
        support = int(true_mask.sum())
        if support == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            supports.append(0)
            continue
        # Recall on this class: fraction of true class-c rows the model
        # labelled as any misbehaviour (not necessarily class c - the held-out
        # class was never trained, so the model can only express it via the
        # "not class 0" decision).
        recall_misbehav = float(((y_pred != 0) & true_mask).sum() / support)
        # Precision on this class: standard.
        if pred_mask.any():
            precision_c = float(((y_pred == c) & true_mask).sum() / pred_mask.sum())
        else:
            precision_c = 0.0
        precisions.append(precision_c)
        recalls.append(recall_misbehav)
        # F1 is undefined when one of the two is 0; report 0.
        if recall_misbehav + precision_c > 0:
            f1s.append(2 * precision_c * recall_misbehav / (precision_c + recall_misbehav))
        else:
            f1s.append(0.0)
        supports.append(support)
    return PerClassReport(
        classes=tuple(classes),
        precision=tuple(precisions),
        recall=tuple(recalls),
        f1=tuple(f1s),
        support=tuple(supports),
    )


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: tuple[int, ...],
    title: str,
    out_path: Path,
) -> None:
    """Save a confusion-matrix heatmap PNG. Matplotlib must be importable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    fig, ax = plt.subplots(figsize=(max(4, len(classes) * 0.5), max(4, len(classes) * 0.5)))
    im = ax.imshow(cm, cmap="Greens", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_recall_bar_plot(
    models: dict[str, PerClassReport],
    title: str,
    out_path: Path,
) -> None:
    """Side-by-side bar chart of per-class recall across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not models:
        return
    classes = next(iter(models.values())).classes
    n_models = len(models)
    n_classes = len(classes)
    width = 0.8 / n_models
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(max(6, n_classes), 4))
    for i, (name, report) in enumerate(models.items()):
        ax.bar(x + i * width - 0.4 + width / 2, report.recall, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("held-out attack class")
    ax.set_ylabel("recall on held-out class (binary)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
