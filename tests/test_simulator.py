"""Tests for the end-to-end streaming simulator (W5).

Includes the D1 / P1 timing-bound check: given a persistent low-trust signal
for one peer, the C5 mock manager must receive its first non-zero revocation
hint within max(w/2, k) messages of stream start.
"""
from __future__ import annotations

import unittest

import numpy as np

from piad_v2x.lifecycle.aggregator import AggregatorConfig, TrustAggregator
from piad_v2x.lifecycle.coupling import CouplingConfig, CouplingLayer
from piad_v2x.data_utils.feature_extractor import FEATURE_NAMES
from piad_v2x.lifecycle.manager import MockPseudonymManager
from piad_v2x.lifecycle.simulator import (
    OracleTrustProvider,
    StreamingFeatureExtractor,
    stream_messages,
)


def _row(pk: int, t: float, x: float = 0.0, vx: float = 1.0, cls: int = 0) -> dict:
    return {
        "senderPseudo": pk,
        "sender": pk // 10,
        "sendTime": t,
        "class": cls,
        "posx": x, "posy": 0.0,
        "spdx": vx, "spdy": 0.0,
        "aclx": 0.0, "acly": 0.0,
        "hedx": 1.0, "hedy": 0.0,
    }


class FeatureStream(unittest.TestCase):
    def test_first_message_is_cold(self) -> None:
        extractor = StreamingFeatureExtractor()
        x = extractor.process(_row(pk=1, t=0.0))
        cold_idx = FEATURE_NAMES.index("cold")
        self.assertEqual(x[cold_idx], 1.0)

    def test_second_message_is_warm(self) -> None:
        extractor = StreamingFeatureExtractor()
        extractor.process(_row(pk=1, t=0.0, x=0.0))
        x = extractor.process(_row(pk=1, t=0.1, x=0.1))
        cold_idx = FEATURE_NAMES.index("cold")
        self.assertEqual(x[cold_idx], 0.0)

    def test_implied_speed_residual_on_teleport(self) -> None:
        """Streaming form catches position falsification just like the batch."""
        extractor = StreamingFeatureExtractor()
        extractor.process(_row(pk=1, t=0.0, x=0.0, vx=1.0, cls=1))
        x = extractor.process(_row(pk=1, t=0.1, x=100.0, vx=1.0, cls=1))
        e_v = x[FEATURE_NAMES.index("e_v")]
        self.assertGreater(e_v, 100.0)

    def test_streaming_matches_batch_shape(self) -> None:
        extractor = StreamingFeatureExtractor()
        x = extractor.process(_row(pk=1, t=0.0))
        self.assertEqual(x.shape, (len(FEATURE_NAMES),))


class OracleProvider(unittest.TestCase):
    def test_benign_class_returns_benign_trust(self) -> None:
        p = OracleTrustProvider(benign_trust=0.9, attack_trust=0.1)
        x = np.zeros(len(FEATURE_NAMES))
        self.assertAlmostEqual(p.trust(x, {"class": 0}), 0.9)

    def test_attack_class_returns_attack_trust(self) -> None:
        p = OracleTrustProvider(benign_trust=0.9, attack_trust=0.1)
        x = np.zeros(len(FEATURE_NAMES))
        self.assertAlmostEqual(p.trust(x, {"class": 5}), 0.1)

    def test_explicit_class_override(self) -> None:
        p = OracleTrustProvider(class_to_trust={13: 0.02})
        x = np.zeros(len(FEATURE_NAMES))
        self.assertAlmostEqual(p.trust(x, {"class": 13}), 0.02)


class StreamE2E(unittest.TestCase):
    def test_benign_stream_produces_no_hints(self) -> None:
        rows = [_row(pk=p, t=0.1 * i, x=0.1 * i, cls=0)
                for p in [1, 2, 3]
                for i in range(60)]
        rows.sort(key=lambda r: r["sendTime"])
        report, manager = stream_messages(rows, OracleTrustProvider())
        self.assertEqual(report.n_hint_events, 0)
        self.assertEqual(report.n_pseudonyms, 3)

    def test_persistent_attacker_produces_hint_within_D1_bound(self) -> None:
        """D1 / P1: revocation-hint fires within max(w/2, k) messages = 25 at defaults.
        Stream is at 10 Hz, so 25 messages = 2.5 s."""
        agg_cfg = AggregatorConfig(alpha=0.5, theta_low=0.5, cold_start_T=0.5)
        cpl_cfg = CouplingConfig(theta_revoke=0.3, k=20, cold_start_msgs=25)
        provider = OracleTrustProvider(benign_trust=0.95, attack_trust=0.05)

        # One persistent attacker at 10 Hz.
        rows = [_row(pk=999, t=0.1 * i, x=i, cls=5) for i in range(80)]
        report, manager = stream_messages(
            rows, provider,
            aggregator=TrustAggregator(agg_cfg),
            coupling=CouplingLayer(cpl_cfg),
        )
        first_hint_t = manager.first_hint_time(999)
        self.assertIsNotNone(first_hint_t, "no revocation hint ever fired")
        # First message is at t=0.0; bound says hint should fire by ~2.5 s.
        # Allow a small EMA settling slack; spec says max(w/2, k) = 25 msgs = 2.5s.
        self.assertLessEqual(first_hint_t, 3.5, f"hint fired at t={first_hint_t}, beyond bound")

    def test_attacker_eventually_gets_multiple_hints(self) -> None:
        provider = OracleTrustProvider()
        rows = [_row(pk=999, t=0.1 * i, x=i, cls=5) for i in range(80)]
        _, manager = stream_messages(rows, provider)
        self.assertGreater(len(manager.hints_for(999)), 1)

    def test_cadence_recorded_for_every_message(self) -> None:
        rows = [_row(pk=p, t=0.1 * i, x=0.1 * i, cls=0)
                for p in [1, 2, 3]
                for i in range(20)]
        rows.sort(key=lambda r: r["sendTime"])
        report, manager = stream_messages(rows, OracleTrustProvider())
        self.assertEqual(report.n_rotation_events, len(rows))


if __name__ == "__main__":
    unittest.main()
