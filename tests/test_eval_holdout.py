"""Tests for W4 held-out attack evaluation utilities."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from piad_v2x.data_utils.dataset import DatasetSpec, split_holdout_attacks
from piad_v2x.models.eval_holdout import (
    binary_report,
    per_class_report,
    to_binary,
)


def _toy_df(n_per_class: int = 40) -> pd.DataFrame:
    """Build a synthetic VeReMi-shaped DataFrame across classes 0, 1, 13, 16."""
    rng = np.random.default_rng(0)
    rows = []
    sender_id = 0
    for cls in [0, 1, 13, 16]:
        for _ in range(n_per_class):
            sender_id += 1
            rows.append({
                "type": 4,
                "sendTime": float(rng.uniform(0.0, 100.0)),
                "sender": sender_id,
                "senderPseudo": sender_id * 10,
                "messageID": rng.integers(0, 1000000),
                "class": cls,
                "posx": float(rng.uniform(0, 1000)),
                "posy": float(rng.uniform(0, 1000)),
                "spdx": float(rng.uniform(-10, 10)),
                "spdy": float(rng.uniform(-10, 10)),
                "aclx": 0.0,
                "acly": 0.0,
                "hedx": 0.0,
                "hedy": 0.0,
                "posx_n": 0.0, "posy_n": 0.0,
                "spdx_n": 0.0, "spdy_n": 0.0,
                "aclx_n": 0.0, "acly_n": 0.0,
                "hedx_n": 0.0, "hedy_n": 0.0,
            })
    return pd.DataFrame(rows)


class HoldoutSplit(unittest.TestCase):
    def test_holdout_classes_absent_from_train(self) -> None:
        df = _toy_df()
        spec = DatasetSpec(n_rows=None, n_splits=3, fold_index=0)
        train_df, test_df = split_holdout_attacks(df, (13, 16), spec)
        self.assertEqual(int((train_df["class"] == 13).sum()), 0)
        self.assertEqual(int((train_df["class"] == 16).sum()), 0)

    def test_test_set_contains_only_benign_and_holdout(self) -> None:
        df = _toy_df()
        spec = DatasetSpec(n_rows=None, n_splits=3, fold_index=0)
        _, test_df = split_holdout_attacks(df, (13, 16), spec)
        allowed = {0, 13, 16}
        self.assertTrue(set(test_df["class"].unique()).issubset(allowed))

    def test_no_sender_overlap(self) -> None:
        df = _toy_df()
        spec = DatasetSpec(n_rows=None, n_splits=3, fold_index=0)
        train_df, test_df = split_holdout_attacks(df, (13,), spec)
        overlap = set(train_df["sender"]) & set(test_df["sender"])
        self.assertEqual(len(overlap), 0)


class BinaryReport(unittest.TestCase):
    def test_to_binary_collapses_non_zero_to_one(self) -> None:
        y = np.array([0, 1, 5, 16, 0])
        self.assertTrue(np.array_equal(to_binary(y), np.array([0, 1, 1, 1, 0])))

    def test_perfect_prediction(self) -> None:
        y_true = np.array([0, 0, 5, 16, 13, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0])  # any non-zero counts as misbehaviour
        rep = binary_report(y_true, y_pred)
        self.assertAlmostEqual(rep.accuracy, 1.0)
        self.assertAlmostEqual(rep.recall_attack, 1.0)
        self.assertAlmostEqual(rep.precision_attack, 1.0)

    def test_misses_all_attacks(self) -> None:
        y_true = np.array([0, 0, 5, 16])
        y_pred = np.array([0, 0, 0, 0])  # everything called benign
        rep = binary_report(y_true, y_pred)
        self.assertAlmostEqual(rep.recall_attack, 0.0)


class PerClassReport(unittest.TestCase):
    def test_recall_credits_any_misbehaviour_label(self) -> None:
        """A model that labels held-out class 13 as 'misbehaviour' (any non-zero)
        should get recall credit, since the held-out class was never trained."""
        y_true = np.array([13, 13, 16, 0, 0])
        y_pred = np.array([5, 7, 1, 0, 0])  # all non-zero predictions count
        rep = per_class_report(y_true, y_pred, (13, 16))
        self.assertEqual(rep.classes, (13, 16))
        self.assertEqual(rep.support, (2, 1))
        self.assertAlmostEqual(rep.recall[0], 1.0)
        self.assertAlmostEqual(rep.recall[1], 1.0)

    def test_zero_recall_when_predicted_benign(self) -> None:
        y_true = np.array([13, 13, 0])
        y_pred = np.array([0, 0, 0])
        rep = per_class_report(y_true, y_pred, (13,))
        self.assertAlmostEqual(rep.recall[0], 0.0)


if __name__ == "__main__":
    unittest.main()
