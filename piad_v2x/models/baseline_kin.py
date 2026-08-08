"""Kinematic-only baseline detector.

This is the *comparator* that the multi-physics PINN must beat. The point
is not to maximise this baseline's performance; it is to give the PINN a fair
and reproducible reference point on the same data, splits, and feature set
(minus the multi-physics terms).

Architecture: scikit-learn RandomForestClassifier on the kinematic
features, with class-weighted training to handle the 45:1 worst-case
imbalance in VeReMi Extension. Multi-class (20-way) by default.

This is not the headline novelty. It is the floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score


@dataclass(frozen=True)
class BaselineConfig:
    n_estimators: int = 200
    max_depth: int | None = 16
    min_samples_leaf: int = 4
    class_weight: str = "balanced"
    random_state: int = 17
    n_jobs: int = -1


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: BaselineConfig | None = None,
) -> RandomForestClassifier:
    cfg = config or BaselineConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    clf.fit(X_train, y_train)
    return clf


@dataclass(frozen=True)
class EvalReport:
    f1_macro: float
    f1_weighted: float
    report: str  # human-readable per-class report

    def short(self) -> str:
        return f"f1_macro={self.f1_macro:.4f} f1_weighted={self.f1_weighted:.4f}"


def evaluate(clf: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> EvalReport:
    preds = clf.predict(X_test)
    return EvalReport(
        f1_macro=f1_score(y_test, preds, average="macro", zero_division=0),
        f1_weighted=f1_score(y_test, preds, average="weighted", zero_division=0),
        report=classification_report(y_test, preds, zero_division=0, digits=3),
    )
