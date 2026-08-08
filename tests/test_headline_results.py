"""Regression guard for the published headline numbers.

Pins the committed result JSONs that feed the published headline numbers, so that a silent change in the
pipeline, or an accidental edit to a committed result file, that would desynchronise
the prose from the evidence fails the test suite. Uses only the standard library
(json) so it adds no dependency. Tolerances are loose: the test guards the claims as
written, not bit-identity.

Each test here skips only if its specific result file is missing from the checkout.

The numbers guarded here:
  - headline: 95.6% accepted-malicious reduction at
    0.20% benign false-revocation, 99.6% isolated, 1.0 s median time-to-isolate.
  - Sybil boundary: reachability gate 0.3%, aggregated-trust gate
    12.5% neutralisation of Sybil pseudonyms.
  - full-corpus closed loop (base kinematic detector): Data Replay Sybil reduces
    only 1.9% and neutralises only 12% of pseudonyms (the honest Sybil-defeats-it cut).
  - scoping negative: physics-in-loss refuted / indistinguishable.
"""

import json
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "results"


def _load(*parts):
    p = RESULTS.joinpath(*parts)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class TestMultiScaleLifecycleHeadline(unittest.TestCase):
    def setUp(self):
        self.d = _load("lifecycle_multiscale.json")
        if self.d is None:
            self.skipTest("lifecycle_multiscale.json not present")

    def test_accepted_malicious_reduction(self):
        r = self.d["accepted_malicious_reduction"]
        self.assertAlmostEqual(r["point"], 0.956, delta=0.002)
        self.assertAlmostEqual(r["ci95"][0], 0.9527, delta=0.002)
        self.assertAlmostEqual(r["ci95"][1], 0.9592, delta=0.002)

    def test_false_revocation(self):
        self.assertAlmostEqual(self.d["false_revocation"]["point"], 0.002, delta=0.0005)

    def test_time_to_isolate(self):
        t = self.d["time_to_isolate"]
        self.assertAlmostEqual(t["frac_isolated"], 0.9959, delta=0.002)
        self.assertAlmostEqual(t["km_median_s_point"], 1.0, delta=0.01)

    def test_detector_flag_rate(self):
        self.assertAlmostEqual(self.d["detector_flag_rate_on_attackers_mean"], 0.8946, delta=0.005)


class TestSybilBoundary(unittest.TestCase):
    def setUp(self):
        self.d = _load("reachability_lifecycle.json")
        if self.d is None:
            self.skipTest("reachability_lifecycle.json not present")

    def test_neutralisation_gates(self):
        n = self.d["neutralised_pct"]
        # reachability gate is precise but low-recall; aggregated-trust gate does better
        self.assertAlmostEqual(n["D_reach"], 0.003, delta=0.001)
        self.assertAlmostEqual(n["D_trust"], 0.125, delta=0.005)
        self.assertGreater(n["D_trust"], n["D_reach"])


class TestFullCorpusClosedLoopSybil(unittest.TestCase):
    """The base-kinematic first cut MUST be the full corpus, not the more
    favourable 1.8M subsample (Data Replay Sybil ~19%/28%)."""

    def setUp(self):
        self.rows = _load("closed_loop.json")
        if self.rows is None:
            self.skipTest("closed_loop.json not present")
        self.by = {r["scenario"]: r for r in self.rows}

    def test_sybil_is_full_corpus_not_subsample(self):
        s = self.by["DataReplaySybil_1416"]
        self.assertAlmostEqual(s["D_vs_C_relative_reduction"], 0.0193, delta=0.01)
        self.assertAlmostEqual(s["frac_attacker_pseudos_neutralised"], 0.1226, delta=0.02)
        # guard against the subsample values silently reappearing
        self.assertLess(s["D_vs_C_relative_reduction"], 0.10,
                        "Sybil reduction looks like the subsample (~19%), not full corpus (~1.9%)")


class TestScopingNegative(unittest.TestCase):
    def setUp(self):
        self.d = _load("placement_bootstrap.json")
        if self.d is None:
            self.skipTest("placement_bootstrap.json not present")

    def test_physics_in_loss_refuted(self):
        self.assertIn("refuted", self.d["verdict_loss_vs_data"].lower())
        self.assertIn("refuted", self.d["verdict_loss_vs_feature"].lower())
        # the loss-vs-data effect is below the minimum detectable effect
        self.assertLess(self.d["loss_vs_data"]["mean"], self.d["MDE"])


if __name__ == "__main__":
    unittest.main()
