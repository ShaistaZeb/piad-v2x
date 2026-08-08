"""VeReMi Extension dataset loader.

Loads the unified VeReMi Extension table (data/veremi.parquet, built by
piad_v2x/tools/build_mixed_table.py; legacy MixAll CSV still accepted) and
provides leakage-safe train/test splits.

Conventions:
- Receiver realism: features are computed per `senderPseudo` (what a real
  receiver sees), not per `sender` (the simulator-internal vehicle id).
- Leakage prevention: train/test split groups by `sender` so the same
  vehicle's trajectories cannot appear in both partitions, even after
  pseudonym rotation. See StratifiedGroupKFold below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Unified single-pipeline table, built by piad_v2x/tools/build_mixed_table.py from the
# per-scenario parquet. (The legacy MixAll CSV path is still accepted by
# load_messages for backward compatibility, but is no longer the default.)
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "veremi.parquet",
)

# Columns we actually use. The simulator-truth columns (without `_n` suffix)
# are at sensible physical scales (m, m/s, m/s², heading-vector components).
# The `_n` columns in this VeReMi Extension release appear to be noise
# *deltas* rather than noisy readings (different absolute scales from the
# truth columns), so the realistic-noise reading is `value + value_n`. For
# we train on the simulator-truth columns directly; a later variant should revisit.
SIM_COLS = [
    "posx", "posy",
    "spdx", "spdy",
    "aclx", "acly",
    "hedx", "hedy",
]
# Carry the noise deltas so a future loader variant can compose noisy readings.
NOISE_DELTA_COLS = [
    "posx_n", "posy_n",
    "spdx_n", "spdy_n",
    "aclx_n", "acly_n",
    "hedx_n", "hedy_n",
]
ID_COLS = ["type", "sendTime", "sender", "senderPseudo", "messageID", "class"]


@dataclass(frozen=True)
class DatasetSpec:
    csv_path: str = DEFAULT_CSV
    n_rows: int | None = None             # subsample for fast iteration
    random_state: int = 17
    n_splits: int = 5                     # for StratifiedGroupKFold
    fold_index: int = 0                   # which fold to use as the test set
    stratify_subsample: bool = True       # keep class proportions when subsampling


def load_messages(spec: DatasetSpec | None = None) -> pd.DataFrame:
    """Load (and optionally subsample) the flattened CSV.

    Returns a DataFrame sorted by (senderPseudo, sendTime) so per-pseudonym
    feature extraction can rely on row order.
    """
    spec = spec or DatasetSpec()
    want = ID_COLS + SIM_COLS + NOISE_DELTA_COLS
    path = str(spec.csv_path)
    if path.endswith((".parquet", ".pq")):
        # Unified single-pipeline table from piad_v2x/tools/build_mixed_table.py. The
        # noise-delta columns are not produced by the raw extractor and are
        # optional (unused by the loss-based detector), so select the intersection.
        df = pd.read_parquet(path)
        df = df[[c for c in want if c in df.columns]]
    else:
        # Legacy flattened CSV path (retained for backward compatibility).
        avail = pd.read_csv(path, nrows=0).columns
        df = pd.read_csv(path, usecols=[c for c in want if c in avail])

    if spec.n_rows is not None and spec.n_rows < len(df):
        if spec.stratify_subsample:
            # Per-class proportional draw so rare attack classes survive.
            # Use explicit per-class sampling (pandas 3 groupby.apply consumes
            # the group key column, so we avoid that path).
            rng = np.random.default_rng(spec.random_state)
            frac = spec.n_rows / len(df)
            parts = []
            for _, g in df.groupby("class", observed=True, sort=False):
                n_target = max(1, int(round(len(g) * frac)))
                parts.append(
                    g.sample(
                        n=min(n_target, len(g)),
                        random_state=int(rng.integers(0, 2**32 - 1)),
                    )
                )
            df = pd.concat(parts, ignore_index=True)
        else:
            df = df.sample(n=spec.n_rows, random_state=spec.random_state)

    df = df.sort_values(["senderPseudo", "sendTime"], kind="stable").reset_index(drop=True)
    return df


def split_by_sender(
    df: pd.DataFrame,
    spec: DatasetSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe split: group by `sender` (vehicle id) so the same vehicle
    cannot straddle train/test. Stratified by attack class on the group level.

    Returns (train_df, test_df).
    """
    spec = spec or DatasetSpec()
    # StratifiedGroupKFold needs per-row y and groups; we pass class as y and
    # sender as the group. Pick spec.fold_index of n_splits as the test fold.
    y = df["class"].to_numpy()
    groups = df["sender"].to_numpy()
    skf = StratifiedGroupKFold(
        n_splits=spec.n_splits, shuffle=True, random_state=spec.random_state
    )
    splits = list(skf.split(np.zeros(len(df)), y, groups))
    train_idx, test_idx = splits[spec.fold_index]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_holdout_attacks(
    df: pd.DataFrame,
    holdout_classes: tuple[int, ...],
    spec: DatasetSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out-attack split.

    Train: rows from train-side senders, with the held-out attack classes
           entirely REMOVED. The model has seen all benign patterns and the
           non-held-out attack patterns.
    Test:  rows from test-side senders, restricted to {benign} ∪ holdout_classes.
           This evaluates whether the model can flag unseen attack patterns
           as misbehaviour (binary detection).

    Senders never overlap across train/test (leakage-safe, same as
    `split_by_sender`).
    """
    spec = spec or DatasetSpec()
    train_df, test_df = split_by_sender(df, spec)
    holdout_set = set(int(c) for c in holdout_classes)

    train_filtered = train_df[~train_df["class"].isin(holdout_set)].reset_index(drop=True)
    test_filtered = test_df[
        test_df["class"].isin(holdout_set) | (test_df["class"] == 0)
    ].reset_index(drop=True)
    return train_filtered, test_filtered


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Convenience: class counts sorted by class id."""
    return df["class"].value_counts().sort_index()


def iter_pseudonym_groups(df: pd.DataFrame) -> Iterable[tuple[int, pd.DataFrame]]:
    """Yield (pseudonym_id, group_dataframe) ordered by sendTime within each group."""
    for pk, g in df.groupby("senderPseudo", sort=False):
        yield int(pk), g
