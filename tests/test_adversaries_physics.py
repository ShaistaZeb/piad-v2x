"""Tests for the Pillar C physics-respecting adversary."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from piad_v2x.attacks.adversaries_physics import (
    fit_benign_speed_gmm,
    make_distribution_matching,
    make_distribution_matching_ar1,
    make_distribution_matching_persistent,
    make_kinematic_capped,
    make_kinematic_respecting,
    rewrite_with_summary,
)


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestKinematicRespecting(unittest.TestCase):

    def test_benign_rows_unchanged(self):
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 10.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 1.0, "posy": 0.0, "spdx": 10.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
        ])
        out = make_kinematic_respecting(df)
        np.testing.assert_array_equal(out["spdx"].values, df["spdx"].values)
        np.testing.assert_array_equal(out["spdy"].values, df["spdy"].values)

    def test_attack_speed_matches_position_delta(self):
        # Posx jumps by 2m in 0.1s, so implied v_x = 20 m/s.
        # Original spdx = 5 m/s (mismatch -> high residual). Rewriting
        # should set spdx to 20.
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 2.0, "posy": 0.0, "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
        ])
        out = make_kinematic_respecting(df)
        # The first row is the "anchor" and cannot be rewritten; the
        # second row should have spdx ~ 20.
        self.assertAlmostEqual(out["spdx"].iloc[1], 20.0, places=4)

    def test_acceleration_rewritten_when_three_points(self):
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 1.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 99.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.2,
             "posx": 4.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 99.0, "acly": 0.0, "class": 13},
        ])
        out = make_kinematic_respecting(df)
        # The middle row's aclx should be rewritten from the (-1,4) position
        # finite difference: a_x = (4 - 2*1 + 0) / 0.1^2 = 200 m/s^2 in idealised form.
        # We just check that it's no longer 99 (i.e. rewriting happened).
        self.assertNotEqual(out["aclx"].iloc[1], 99.0)

    def test_target_classes_filter_isolates_chosen_attacks(self):
        # Class 13 is target; class 14 is NOT.
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 5.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 2, "senderPseudo": "B", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 9.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 14},
            {"sender": 2, "senderPseudo": "B", "sendTime": 0.1,
             "posx": 7.0, "posy": 0.0, "spdx": 9.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 14},
        ])
        out = make_kinematic_respecting(df, target_classes=(13,))
        # Class 13 row 1 was rewritten.
        self.assertAlmostEqual(out["spdx"].iloc[1], 50.0, places=4)
        # Class 14 row 3 was NOT.
        self.assertAlmostEqual(out["spdx"].iloc[3], 9.0, places=4)

    def test_rewrite_summary_counts(self):
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 1.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.2,
             "posx": 2.0, "posy": 0.0, "spdx": 1.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
        ])
        new, summary = rewrite_with_summary(df, target_classes=(13,))
        self.assertEqual(summary.n_total, 3)
        self.assertEqual(summary.n_attack, 2)
        # Row 0 cannot be rewritten (no prior); row 1 can.
        self.assertEqual(summary.n_rewritten, 1)
        self.assertEqual(summary.n_unchanged_attack, 1)

    def test_kinematic_capped_constrains_speed_magnitude(self):
        # Position jumps by 100m in 0.1s -> implied speed 1000 m/s.
        # With v_max=30, the cap is 3m -> new position = prev + 3m,
        # implied speed = 30 m/s.
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 100.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
        ])
        out = make_kinematic_capped(df, v_max=30.0, target_classes=(13,))
        # Cap is 3m per step, so new position should be 3m, not 100m.
        self.assertAlmostEqual(out["posx"].iloc[1], 3.0, places=4)
        # Implied speed = 3 / 0.1 = 30 m/s, within v_max.
        self.assertAlmostEqual(out["spdx"].iloc[1], 30.0, places=4)
        self.assertLessEqual(abs(out["spdx"].iloc[1]), 30.0)

    def test_kinematic_capped_leaves_small_deviations_alone(self):
        # Position jumps by 2m in 0.1s -> implied speed 20 m/s, below v_max=30.
        # Should be passed through unchanged.
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 2.0, "posy": 0.0, "spdx": 99.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
        ])
        out = make_kinematic_capped(df, v_max=30.0, target_classes=(13,))
        # Position unchanged.
        self.assertAlmostEqual(out["posx"].iloc[1], 2.0, places=4)
        # Speed becomes implied speed (20 m/s), not the original 99.
        self.assertAlmostEqual(out["spdx"].iloc[1], 20.0, places=4)

    def test_kinematic_capped_benign_unchanged(self):
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 1000.0, "posy": 0.0, "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
        ])
        out = make_kinematic_capped(df, v_max=30.0, target_classes=(13,))
        # Benign row is left alone even with absurd position.
        self.assertAlmostEqual(out["posx"].iloc[1], 1000.0, places=4)
        self.assertAlmostEqual(out["spdx"].iloc[1], 5.0, places=4)

    def test_distribution_matching_changes_attack_features(self):
        # Synthetic benign: speeds around (15, 0) m/s.
        rng = np.random.default_rng(0)
        benign_rows = []
        for k in range(200):
            benign_rows.append({
                "sender": k + 100, "senderPseudo": f"B{k}",
                "sendTime": 0.0,
                "posx": 0.0, "posy": 0.0,
                "spdx": 15.0 + rng.normal() * 0.5,
                "spdy": 0.0 + rng.normal() * 0.5,
                "aclx": rng.normal() * 0.2,
                "acly": rng.normal() * 0.2,
                "class": 0,
            })
        benign_df = _build_df(benign_rows)
        gmm = fit_benign_speed_gmm(benign_df, n_components=2, random_state=0)

        # Attack: extreme spdx, spdy
        attack_df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0,
             "spdx": 100.0, "spdy": 100.0,
             "aclx": 50.0, "acly": 50.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 10.0, "posy": 10.0,
             "spdx": 100.0, "spdy": 100.0,
             "aclx": 50.0, "acly": 50.0, "class": 13},
        ])
        out = make_distribution_matching(attack_df, gmm, target_classes=(13,))
        # Rewritten spdx (row 1) should be near benign mean (15), not 100.
        self.assertLess(abs(out["spdx"].iloc[1] - 15.0), 5.0)
        # The first row is the anchor; it stays at original 100.
        self.assertAlmostEqual(out["spdx"].iloc[0], 100.0, places=4)

    def test_persistent_matching_keeps_within_vehicle_variance_low(self):
        """A-PHY-4: same-pseudonym attack messages should have lower
        spdx variance than A-PHY-3 (i.i.d. mixture sampling) because
        all draws come from the SAME Gaussian."""
        rng = np.random.default_rng(0)
        benign_rows = []
        for k in range(200):
            # Two well-separated speed clusters: {15 m/s, -15 m/s}
            cluster_id = k % 2
            mu_x = 15.0 if cluster_id == 0 else -15.0
            benign_rows.append({
                "sender": k + 100, "senderPseudo": f"B{k}",
                "sendTime": 0.0,
                "posx": 0.0, "posy": 0.0,
                "spdx": mu_x + rng.normal() * 0.3,
                "spdy": 0.0 + rng.normal() * 0.3,
                "aclx": 0.0, "acly": 0.0, "class": 0,
            })
        benign_df = _build_df(benign_rows)
        gmm = fit_benign_speed_gmm(benign_df, n_components=2, random_state=0)

        # An attack pseudonym with many messages.
        attack_rows = []
        for k in range(20):
            attack_rows.append({
                "sender": 9, "senderPseudo": "A",
                "sendTime": 0.1 * k,
                "posx": 0.0, "posy": 0.0,
                "spdx": 0.0, "spdy": 0.0,
                "aclx": 0.0, "acly": 0.0, "class": 13,
            })
        attack_df = _build_df(attack_rows)

        # A-PHY-3 mixes both components within this single vehicle's stream.
        out3 = make_distribution_matching(attack_df, gmm,
                                            target_classes=(13,), seed=0)
        var3 = float(np.var(out3["spdx"].iloc[1:].values))

        # A-PHY-4 sticks to one component; within-vehicle variance should be
        # SMALL (~0.3^2 = 0.09 for one cluster) regardless of which one chosen.
        out4 = make_distribution_matching_persistent(
            attack_df, gmm, target_classes=(13,), seed=0,
        )
        var4 = float(np.var(out4["spdx"].iloc[1:].values))

        # A-PHY-4 variance should be MUCH lower than A-PHY-3 (mixing two
        # clusters spread 30 m/s apart inflates the variance).
        self.assertLess(var4, var3 * 0.5)

    def test_ar1_matching_reduces_first_order_differences(self):
        """A-PHY-5: AR(1)-smoothed sampling should produce LOWER
        consecutive speed differences than A-PHY-3 i.i.d. sampling
        when alpha is high."""
        rng = np.random.default_rng(0)
        benign_rows = []
        for k in range(200):
            cluster_id = k % 2
            mu_x = 15.0 if cluster_id == 0 else -15.0
            benign_rows.append({
                "sender": k + 100, "senderPseudo": f"B{k}",
                "sendTime": 0.0,
                "posx": 0.0, "posy": 0.0,
                "spdx": mu_x + rng.normal() * 0.3,
                "spdy": rng.normal() * 0.3,
                "aclx": 0.0, "acly": 0.0, "class": 0,
            })
        benign_df = _build_df(benign_rows)
        gmm = fit_benign_speed_gmm(benign_df, n_components=2, random_state=0)

        attack_rows = [
            {"sender": 9, "senderPseudo": "A",
             "sendTime": 0.1 * k,
             "posx": 0.0, "posy": 0.0,
             "spdx": 0.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13}
            for k in range(40)
        ]
        attack_df = _build_df(attack_rows)

        out3 = make_distribution_matching(attack_df, gmm,
                                            target_classes=(13,), seed=0)
        # |delta spdx|: mean absolute consecutive difference
        diff3 = np.mean(np.abs(np.diff(out3["spdx"].iloc[1:].values)))

        out5 = make_distribution_matching_ar1(
            attack_df, gmm, alpha=0.9, target_classes=(13,), seed=0,
        )
        diff5 = np.mean(np.abs(np.diff(out5["spdx"].iloc[1:].values)))

        # AR(1)-smoothed differences should be MUCH smaller.
        self.assertLess(diff5, diff3 * 0.5)

    def test_distribution_matching_leaves_benign_unchanged(self):
        rng = np.random.default_rng(0)
        benign_rows = []
        for k in range(100):
            benign_rows.append({
                "sender": k + 100, "senderPseudo": f"B{k}",
                "sendTime": 0.0,
                "posx": 0.0, "posy": 0.0,
                "spdx": 15.0 + rng.normal() * 0.5,
                "spdy": 0.0 + rng.normal() * 0.5,
                "aclx": 0.0, "acly": 0.0, "class": 0,
            })
        benign_df = _build_df(benign_rows)
        gmm = fit_benign_speed_gmm(benign_df, n_components=1, random_state=0)
        df = _build_df([
            {"sender": 2, "senderPseudo": "X", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0,
             "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
            {"sender": 2, "senderPseudo": "X", "sendTime": 0.1,
             "posx": 0.5, "posy": 0.0,
             "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 0},
        ])
        out = make_distribution_matching(df, gmm, target_classes=(13,))
        # Benign rows unchanged.
        self.assertAlmostEqual(out["spdx"].iloc[1], 5.0, places=4)
        self.assertAlmostEqual(out["posx"].iloc[1], 0.5, places=4)

    def test_does_not_modify_input_df(self):
        df = _build_df([
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.0,
             "posx": 0.0, "posy": 0.0, "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
            {"sender": 1, "senderPseudo": "A", "sendTime": 0.1,
             "posx": 2.0, "posy": 0.0, "spdx": 5.0, "spdy": 0.0,
             "aclx": 0.0, "acly": 0.0, "class": 13},
        ])
        original_spdx = df["spdx"].copy()
        _ = make_kinematic_respecting(df)
        np.testing.assert_array_equal(df["spdx"].values, original_spdx.values)


if __name__ == "__main__":
    unittest.main()
