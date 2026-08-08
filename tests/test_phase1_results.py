"""Regression guard for the LEGACY Phase 1 VeReMi result set.

NOTE (2026-07-19): these numbers are the retired legacy pipeline
(experiments/results/phase1_veremi_eval/, DD-007 modulators), NOT the current
dissertation/paper headline. The current headline (95.6% accepted-malicious reduction,
etc.) is guarded by `test_headline_results.py`. This file is retained only so the
legacy claims stay pinned where the legacy results are still present; it skips cleanly
otherwise. Do not treat a pass here as covering the current claims.

Pins the committed result CSVs (experiments/results/phase1_veremi_eval/) so that a
silent change in the pipeline that alters the headline claims fails the test suite.
Uses only the standard library (csv) so it adds no dependency. Tolerances are loose
because the numbers are 5-fold means; the test guards the claims, not bit-identity.

Added 2026-06-22 per the reproducibility audit (HIGH: no test asserted any headline
number). Skips cleanly if the gitignored results are not present in the checkout.
"""

import csv
import unittest
from pathlib import Path

RESULTS = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "results"
    / "phase1_veremi_eval"
)


def _rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPhase1HeadlineResults(unittest.TestCase):
    def setUp(self):
        if not (RESULTS / "summary.csv").exists():
            self.skipTest("phase1_veremi_eval results not present (gitignored)")
        self.summary = {r["variant"]: r for r in _rows("summary.csv")}

    def test_closed_loop_recall_at_zero_fp(self):
        cl = self.summary["closed_loop"]
        self.assertAlmostEqual(float(cl["tp_rate_mean"]), 0.0715, delta=0.005)
        self.assertEqual(float(cl["fp_rate_mean"]), 0.0)

    def test_naive_higher_recall_but_nonzero_fp(self):
        nv = self.summary["naive_revocation"]
        self.assertAlmostEqual(float(nv["tp_rate_mean"]), 0.635, delta=0.02)
        self.assertGreater(float(nv["fp_rate_mean"]), 0.02)

    def test_decoupled_baselines_catch_nothing(self):
        for v in ("pseudonym_only", "detection_only"):
            self.assertEqual(float(self.summary[v]["tp_rate_mean"]), 0.0)

    def test_closed_loop_indistinguishable_from_trust_only(self):
        # DD-007 multiplicative modulators add no measurable detection on VeReMi.
        sig = {(r["variant_a"], r["variant_b"]): r for r in _rows("significance.csv")}
        row = sig.get(("closed_loop", "trust_only"))
        self.assertIsNotNone(row, "missing closed_loop vs trust_only significance row")
        self.assertEqual(row["verdict"], "indistinguishable")
        self.assertAlmostEqual(float(row["mean_diff"]), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
