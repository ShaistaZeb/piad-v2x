"""Tests for C1 feature extractor (W2)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from piad_v2x.data_utils.feature_extractor import FEATURE_NAMES, extract_features


def _make_messages(rows: list[dict]) -> pd.DataFrame:
    # Fill any missing columns with 0.0; tests only set what they need.
    cols = [
        "type", "sendTime", "sender", "senderPseudo", "messageID", "class",
        "posx", "posy", "spdx", "spdy",
        "aclx", "acly", "hedx", "hedy",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    # Per-row omitted fields become NaN in a mixed-dict frame; fill them so
    # "tests only set what they need" holds for every column, including the
    # instantaneous plausibility flags computed on every row.
    df = df.fillna(0.0)
    return df[cols].sort_values(["senderPseudo", "sendTime"]).reset_index(drop=True)


class FeatureShape(unittest.TestCase):
    def test_feature_vector_length_matches_names(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 0,
             "posx": 0.1, "posy": 0.0, "spdx": 1.0, "spdy": 0.0},
        ])
        batch = extract_features(df)
        self.assertEqual(batch.X.shape, (2, len(FEATURE_NAMES)))
        self.assertEqual(batch.y.shape, (2,))


class ColdStartHandling(unittest.TestCase):
    def test_first_message_is_cold(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "spdx": 1.0, "spdy": 0.0},
        ])
        batch = extract_features(df)
        cold_idx = FEATURE_NAMES.index("cold")
        self.assertEqual(batch.X[0, cold_idx], 1.0)

    def test_pseudonym_rotation_resets_cold(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0, "posy": 0.0, "spdx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 0,
             "posx": 0.1, "posy": 0.0, "spdx": 1.0},
            # Pseudonym rotates: first message under the new id is cold.
            {"sender": 1, "senderPseudo": 200, "sendTime": 0.2, "class": 0,
             "posx": 0.2, "posy": 0.0, "spdx": 1.0},
        ])
        batch = extract_features(df)
        cold_col = FEATURE_NAMES.index("cold")
        # After sort by (senderPseudo, sendTime): rows of pseudo 100, then 200.
        # The first message of EACH pseudonym should be cold.
        cold_by_pseudo = {}
        for i, pk in enumerate(batch.pseudonyms):
            cold_by_pseudo.setdefault(int(pk), []).append(batch.X[i, cold_col])
        self.assertEqual(cold_by_pseudo[100][0], 1.0)
        self.assertEqual(cold_by_pseudo[200][0], 1.0)
        # Second message of pseudo 100 is warm.
        self.assertEqual(cold_by_pseudo[100][1], 0.0)

    def test_kinematic_residuals_masked_on_cold(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "spdx": 50.0},
        ])
        batch = extract_features(df)
        e_v = FEATURE_NAMES.index("e_v")
        self.assertEqual(batch.X[0, e_v], 0.0)

    def test_large_gap_marks_cold(self) -> None:
        """A 10-second gap between messages of the same pseudonym is treated
        as cold-start; deltas across it are unreliable."""
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 10.0, "class": 0,
             "posx": 50.0},
        ])
        batch = extract_features(df)
        cold_col = FEATURE_NAMES.index("cold")
        self.assertEqual(batch.X[1, cold_col], 1.0)


class KinematicResiduals(unittest.TestCase):
    def test_implied_speed_matches_reported_when_consistent(self) -> None:
        # Two messages 0.1 s apart, vehicle moves 0.1m in x at reported speed 1.0 m/s.
        # Implied v* = 0.1 / 0.1 = 1.0; reported |spd| = 1.0; e_v = 0.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0, "spdx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 0,
             "posx": 0.1, "spdx": 1.0},
        ])
        batch = extract_features(df)
        e_v = FEATURE_NAMES.index("e_v")
        self.assertAlmostEqual(batch.X[1, e_v], 0.0, places=6)

    def test_implied_speed_diverges_when_position_lies(self) -> None:
        # Vehicle "teleports" 100 m in 0.1 s while reporting 1 m/s.
        # Implied v* = 1000 m/s; e_v = 1000 - 1 = large positive.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "posx": 0.0, "spdx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 1,
             "posx": 100.0, "spdx": 1.0},
        ])
        batch = extract_features(df)
        e_v = FEATURE_NAMES.index("e_v")
        self.assertGreater(batch.X[1, e_v], 100.0)


class PlausibilityFlags(unittest.TestCase):
    """S1 VCADS-style physics-plausibility features."""

    def _col(self, batch, name):
        return batch.X[:, FEATURE_NAMES.index(name)]

    def test_all_flags_in_unit_interval(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0, "spdx": 1.0, "hedx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 1,
             "posx": 100.0, "spdx": 1.0, "aclx": 50.0, "hedx": -1.0},
        ])
        batch = extract_features(df)
        for name in ("p_spd_lim", "p_acl_lim", "p_posjump", "p_spd_cons", "p_hed_cons"):
            col = self._col(batch, name)
            self.assertTrue(np.all(col >= 0.0) and np.all(col <= 1.0), name)

    def test_speed_limit_flag_is_instantaneous(self) -> None:
        # Single cold message reporting 100 m/s (> 70 default) must still flag.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "spdx": 100.0},
        ])
        batch = extract_features(df)
        self.assertGreater(self._col(batch, "p_spd_lim")[0], 0.0)
        # Cold row, but speed flag does not depend on a prior message.
        self.assertEqual(self._col(batch, "cold")[0], 1.0)

    def test_accel_limit_flag(self) -> None:
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "aclx": 20.0},  # > 10 default
        ])
        batch = extract_features(df)
        self.assertEqual(self._col(batch, "p_acl_lim")[0], 1.0)

    def test_posjump_flag_fires_on_teleport_and_is_cold_safe(self) -> None:
        # Teleport 100 m in 0.1 s -> implied speed 1000 m/s >> v_max.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "posx": 0.0, "spdx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 1,
             "posx": 100.0, "spdx": 1.0},
        ])
        batch = extract_features(df)
        pj = self._col(batch, "p_posjump")
        self.assertEqual(pj[0], 0.0)        # cold row: masked
        self.assertEqual(pj[1], 1.0)        # warm row: full violation

    def test_speed_consistency_flag(self) -> None:
        # Same teleport: |e_v| huge -> p_spd_cons saturates on the warm row.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "posx": 0.0, "spdx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 1,
             "posx": 100.0, "spdx": 1.0},
        ])
        batch = extract_features(df)
        self.assertEqual(self._col(batch, "p_spd_cons")[1], 1.0)

    def test_heading_consistency_flag(self) -> None:
        # Moving +x at 50 m/s but reporting heading -x -> cos = -1 -> flag = 1.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 1,
             "posx": 0.0, "hedx": -1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 1,
             "posx": 5.0, "hedx": -1.0},
        ])
        batch = extract_features(df)
        hc = self._col(batch, "p_hed_cons")
        self.assertEqual(hc[0], 0.0)                 # cold: masked
        self.assertAlmostEqual(hc[1], 1.0, places=5)  # opposed heading

    def test_benign_consistent_motion_flags_zero(self) -> None:
        # Smooth motion: 1 m in 0.1 s = 10 m/s, reported 10 m/s, heading +x.
        df = _make_messages([
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.0, "class": 0,
             "posx": 0.0, "spdx": 10.0, "hedx": 1.0},
            {"sender": 1, "senderPseudo": 100, "sendTime": 0.1, "class": 0,
             "posx": 1.0, "spdx": 10.0, "hedx": 1.0},
        ])
        batch = extract_features(df)
        for name in ("p_spd_lim", "p_acl_lim", "p_posjump", "p_spd_cons", "p_hed_cons"):
            self.assertAlmostEqual(self._col(batch, name)[1], 0.0, places=5, msg=name)


class GroupIdentity(unittest.TestCase):
    def test_senders_and_pseudonyms_propagated(self) -> None:
        df = _make_messages([
            {"sender": 7, "senderPseudo": 700, "sendTime": 0.0, "class": 0},
            {"sender": 7, "senderPseudo": 700, "sendTime": 0.1, "class": 0},
        ])
        batch = extract_features(df)
        self.assertTrue(np.all(batch.senders == 7))
        self.assertTrue(np.all(batch.pseudonyms == 700))


if __name__ == "__main__":
    unittest.main()
