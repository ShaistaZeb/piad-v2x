"""Tests for the coupling comparators + the frame-event stream."""
from __future__ import annotations

import unittest

from piad_v2x.attacks.adversaries import (
    honest_traffic,
    merge_streams,
    persistent_attacker,
    stealthy_attacker,
)
from piad_v2x.models.comparators import (
    COUPLING_VARIANTS,
    ChurnHintConfig,
    ContextOnlyCoupling,
    DensityHintConfig,
    ChurnHintCoupling,
    DensityHintCoupling,
    NaiveThresholdConfig,
    NaiveThresholdRevocation,
    NoCoupling,
    TrustOnlyCoupling,
)
from piad_v2x.lifecycle.coupling import NeighbourhoodSummary, PeerState
from piad_v2x.lifecycle.simulator import OracleTrustProvider, stream_messages


class ContextOnly(unittest.TestCase):
    def test_cadence_increases_with_density(self) -> None:
        c = ContextOnlyCoupling()
        low = NeighbourhoodSummary(mean=0.9, q25=0.85, n=2, rho_t=2.0)
        high = NeighbourhoodSummary(mean=0.9, q25=0.85, n=30, rho_t=30.0)
        self.assertGreater(c.cadence(high), c.cadence(low))

    def test_no_hints_emitted(self) -> None:
        """SAPACS-style does not consume detector output, so no hints."""
        c = ContextOnlyCoupling()
        peer = PeerState(msg_count=100, T_peer=0.01, count_below=80)
        self.assertEqual(c.revocation_hint(peer), 0.0)

    def test_cadence_independent_of_trust(self) -> None:
        c = ContextOnlyCoupling()
        same_density = lambda mean: NeighbourhoodSummary(
            mean=mean, q25=mean, n=10, rho_t=10.0
        )
        self.assertAlmostEqual(c.cadence(same_density(0.9)),
                               c.cadence(same_density(0.1)))


class TrustOnly(unittest.TestCase):
    def test_cadence_rises_when_trust_drops(self) -> None:
        c = TrustOnlyCoupling()
        healthy = NeighbourhoodSummary(mean=0.9, q25=0.8, n=10)
        hostile = NeighbourhoodSummary(mean=0.1, q25=0.05, n=10)
        self.assertGreater(c.cadence(hostile), c.cadence(healthy))

    def test_cadence_ignores_density(self) -> None:
        """No g_density: cadence response is independent of rho_t."""
        c = TrustOnlyCoupling()
        sparse = NeighbourhoodSummary(mean=0.1, q25=0.05, n=10, rho_t=3.0)
        dense = NeighbourhoodSummary(mean=0.1, q25=0.05, n=10, rho_t=30.0)
        self.assertAlmostEqual(c.cadence(sparse), c.cadence(dense))

    def test_hint_surface_matches_full_framework(self) -> None:
        c = TrustOnlyCoupling()
        bad_peer = PeerState(msg_count=100, T_peer=0.05, count_below=80)
        self.assertGreater(c.revocation_hint(bad_peer), 0.0)


class NoCouplingFloor(unittest.TestCase):
    def test_cadence_is_constant(self) -> None:
        c = NoCoupling()
        n1 = NeighbourhoodSummary(mean=0.9, q25=0.8, n=10, rho_t=2.0)
        n2 = NeighbourhoodSummary(mean=0.1, q25=0.05, n=20, rho_t=30.0)
        self.assertAlmostEqual(c.cadence(n1), c.cadence(n2))

    def test_no_hints(self) -> None:
        c = NoCoupling()
        bad_peer = PeerState(msg_count=100, T_peer=0.05, count_below=80)
        self.assertEqual(c.revocation_hint(bad_peer), 0.0)


class Registry(unittest.TestCase):
    def test_baseline_four_variants_present(self) -> None:
        # The four baseline variants must remain present and unchanged;
        # adding a variant must not remove any.
        self.assertTrue({
            "no_coupling", "context_only", "trust_only", "full_coupling",
        }.issubset(set(COUPLING_VARIANTS.keys())))

    def test_all_variants_have_cadence_and_revocation_hint(self) -> None:
        for name, cls in COUPLING_VARIANTS.items():
            instance = cls()
            self.assertTrue(hasattr(instance, "cadence"),
                            f"{name} missing cadence()")
            self.assertTrue(hasattr(instance, "revocation_hint"),
                            f"{name} missing revocation_hint()")


class FrameEventStream(unittest.TestCase):
    def test_persistent_attacker_emits_frame_events(self) -> None:
        """Persistent attacker emits class != 0 every message -> every frame
        should trip the F1 frame_event_threshold."""
        rows = persistent_attacker(pk=999, duration_s=2.0)
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
            frame_event_threshold=0.2,
        )
        self.assertEqual(report.n_frame_events, len(rows))

    def test_stealth_attacker_emits_frame_events_for_attack_bursts(self) -> None:
        """Stealth attacker (3 attack / 7 benign cycles): R_p path won't fire
        but each attack frame should still emit a frame event."""
        rows = stealthy_attacker(pk=999, duration_s=2.0,
                                 burst_attack=3, burst_benign=7)
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
            frame_event_threshold=0.2,
        )
        # 30% attack rate -> ~30% of frames flagged.
        attack_count = sum(1 for r in rows if r["class"] != 0)
        self.assertEqual(report.n_frame_events, attack_count)
        # Cumulative R_p hint path should remain quiet for stealth.
        self.assertEqual(report.n_hint_events, 0)

    def test_benign_traffic_produces_no_frame_events(self) -> None:
        rows = honest_traffic(n_peers=5, duration_s=5.0)
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
            frame_event_threshold=0.2,
        )
        self.assertEqual(report.n_frame_events, 0)

    def test_frame_event_threshold_disabled_by_default(self) -> None:
        rows = persistent_attacker(pk=999, duration_s=2.0)
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
        )
        self.assertEqual(report.n_frame_events, 0)


class FullStackWithComparator(unittest.TestCase):
    def test_context_only_in_simulator_emits_no_hints(self) -> None:
        rows = merge_streams(
            honest_traffic(n_peers=3, duration_s=5.0),
            persistent_attacker(pk=9999, duration_s=5.0),
        )
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
            coupling=ContextOnlyCoupling(),
        )
        self.assertEqual(report.n_hint_events, 0)

    def test_trust_only_in_simulator_emits_hints_for_persistent(self) -> None:
        rows = merge_streams(
            honest_traffic(n_peers=3, duration_s=8.0),
            persistent_attacker(pk=9999, duration_s=8.0),
        )
        report, _ = stream_messages(
            rows,
            OracleTrustProvider(benign_trust=0.95, attack_trust=0.05),
            coupling=TrustOnlyCoupling(),
        )
        self.assertGreater(report.n_hint_events, 0)


class ChurnHintVariant(unittest.TestCase):
    """Trust-independent churn-driven hint path."""

    def _cold_peer(self) -> PeerState:
        # Fresh peer, high trust, no consecutive-low count - would NOT
        # produce a trust-based hint.
        return PeerState(msg_count=5, T_peer=0.99, count_below=0)

    def _settled_peer(self) -> PeerState:
        return PeerState(msg_count=200, T_peer=0.99, count_below=0)

    def test_churn_hint_fires_for_cold_peer_in_high_churn(self) -> None:
        c = ChurnHintCoupling()
        nbh = NeighbourhoodSummary(
            mean=0.95, q25=0.90, n=20, rho_t=20.0, kappa_t=10.0
        )
        self.assertGreater(c.churn_hint(self._cold_peer(), nbh), 0.0)

    def test_churn_hint_silent_for_settled_peer(self) -> None:
        c = ChurnHintCoupling()
        nbh = NeighbourhoodSummary(
            mean=0.95, q25=0.90, n=20, rho_t=20.0, kappa_t=10.0
        )
        self.assertEqual(c.churn_hint(self._settled_peer(), nbh), 0.0)

    def test_churn_hint_silent_below_kappa_threshold(self) -> None:
        c = ChurnHintCoupling(
            churn_config=ChurnHintConfig(kappa_threshold=3.0)
        )
        nbh = NeighbourhoodSummary(
            mean=0.95, q25=0.90, n=20, rho_t=20.0, kappa_t=1.0
        )
        self.assertEqual(c.churn_hint(self._cold_peer(), nbh), 0.0)

    def test_churn_hint_independent_of_trust(self) -> None:
        """Kinematic-respecting case: per-peer trust looks benign, but churn still fires."""
        c = ChurnHintCoupling()
        nbh = NeighbourhoodSummary(
            mean=1.0, q25=1.0, n=20, rho_t=20.0, kappa_t=10.0
        )
        benign_looking = PeerState(msg_count=5, T_peer=1.0, count_below=0)
        self.assertGreater(c.churn_hint(benign_looking, nbh), 0.0)

    def test_cadence_unchanged_from_full_coupling(self) -> None:
        """The new variant inherits cadence(); regression check."""
        from piad_v2x.lifecycle.coupling import CouplingLayer
        base = CouplingLayer()
        ext = ChurnHintCoupling()
        for kappa in (0.0, 1.0, 5.0, 50.0):
            nbh = NeighbourhoodSummary(
                mean=0.4, q25=0.3, n=10, rho_t=15.0, kappa_t=kappa
            )
            self.assertAlmostEqual(base.cadence(nbh), ext.cadence(nbh))

    def test_registered_in_coupling_variants(self) -> None:
        self.assertIn("full_churn_hint", COUPLING_VARIANTS)
        self.assertIs(COUPLING_VARIANTS["full_churn_hint"], ChurnHintCoupling)


class DensityHintVariant(unittest.TestCase):
    """Trust-independent cumulative-density hint path."""

    def _nbh(self) -> NeighbourhoodSummary:
        return NeighbourhoodSummary(
            mean=1.0, q25=1.0, n=10, rho_t=20.0, kappa_t=0.5
        )

    def _peer(self, msg_count: int = 5) -> PeerState:
        return PeerState(msg_count=msg_count, T_peer=1.0, count_below=0)

    def test_silent_below_unique_threshold(self) -> None:
        c = DensityHintCoupling(
            density_config=DensityHintConfig(unique_threshold=8)
        )
        for i in range(5):  # 5 unique peers seen
            c.density_hint(self._peer(), self._nbh(), pk=f"pk{i}", now=float(i))
        # 6th unique peer at t=10 - still below threshold (8)
        h = c.density_hint(self._peer(), self._nbh(), pk="pk5", now=10.0)
        self.assertEqual(h, 0.0)

    def test_fires_above_unique_threshold_for_recent_peer(self) -> None:
        c = DensityHintCoupling(
            density_config=DensityHintConfig(
                unique_threshold=5, recent_window_s=10.0
            )
        )
        # Seed 5 peers at t=0..4
        for i in range(5):
            c.density_hint(self._peer(), self._nbh(), pk=f"pk{i}", now=float(i))
        # 9th peer at t=12 -> unique count = 6, above threshold=5,
        # and this peer is recent.
        h = c.density_hint(self._peer(), self._nbh(), pk="pk5", now=12.0)
        self.assertGreater(h, 0.0)

    def test_silent_for_settled_peer_outside_recent_window(self) -> None:
        c = DensityHintCoupling(
            density_config=DensityHintConfig(
                unique_threshold=3, recent_window_s=5.0
            )
        )
        # Seed an early peer
        c.density_hint(self._peer(), self._nbh(), pk="early", now=0.0)
        # Add enough peers to exceed threshold
        for i in range(5):
            c.density_hint(self._peer(), self._nbh(), pk=f"new{i}",
                           now=20.0 + i)
        # Re-query the early peer at t=25 -> settled, outside recent window.
        h = c.density_hint(self._peer(), self._nbh(), pk="early", now=25.0)
        self.assertEqual(h, 0.0)

    def test_independent_of_trust_and_kappa(self) -> None:
        """Kinematic-respecting + slow-stagger case: trust=1.0, kappa_t low, but density
        signal still fires."""
        c = DensityHintCoupling(
            density_config=DensityHintConfig(unique_threshold=3)
        )
        nbh_low_kappa = NeighbourhoodSummary(
            mean=1.0, q25=1.0, n=10, rho_t=20.0, kappa_t=0.1
        )
        benign_looking = PeerState(msg_count=2, T_peer=1.0, count_below=0)
        for i in range(4):  # 4 peers - exceed threshold
            c.density_hint(benign_looking, nbh_low_kappa,
                           pk=f"pk{i}", now=float(i))
        # Newest peer is fresh and unique-count is 4 > 3 -> fires.
        h = c.density_hint(benign_looking, nbh_low_kappa,
                           pk="pk3", now=3.0)
        self.assertGreater(h, 0.0)

    def test_cadence_and_churn_unchanged_from_parent(self) -> None:
        base = ChurnHintCoupling()
        ext = DensityHintCoupling()
        for kappa in (0.0, 1.0, 5.0):
            nbh = NeighbourhoodSummary(
                mean=0.4, q25=0.3, n=10, rho_t=15.0, kappa_t=kappa
            )
            self.assertAlmostEqual(base.cadence(nbh), ext.cadence(nbh))
            peer = PeerState(msg_count=5, T_peer=0.99, count_below=0)
            self.assertAlmostEqual(
                base.churn_hint(peer, nbh), ext.churn_hint(peer, nbh)
            )

    def test_registered_in_coupling_variants(self) -> None:
        self.assertIn("full_density_hint", COUPLING_VARIANTS)
        self.assertIs(
            COUPLING_VARIANTS["full_density_hint"], DensityHintCoupling
        )


class NaiveThresholdRevocationVariant(unittest.TestCase):
    """Detection-only baseline: instant trust-threshold revocation."""

    def test_fires_below_threshold(self) -> None:
        c = NaiveThresholdRevocation(
            config=NaiveThresholdConfig(revoke_threshold=0.5)
        )
        low_peer = PeerState(msg_count=1, T_peer=0.1, count_below=0)
        self.assertEqual(c.revocation_hint(low_peer), 1.0)

    def test_silent_above_threshold(self) -> None:
        c = NaiveThresholdRevocation(
            config=NaiveThresholdConfig(revoke_threshold=0.5)
        )
        high_peer = PeerState(msg_count=1, T_peer=0.9, count_below=0)
        self.assertEqual(c.revocation_hint(high_peer), 0.0)

    def test_no_cold_start_gate(self) -> None:
        """Fires even on msg_count=1, unlike trust_only / full_coupling."""
        c = NaiveThresholdRevocation()
        fresh_peer = PeerState(msg_count=1, T_peer=0.1, count_below=0)
        self.assertGreater(c.revocation_hint(fresh_peer), 0.0)

    def test_no_persistence_requirement(self) -> None:
        """Fires even with count_below=0 (no persistence required)."""
        c = NaiveThresholdRevocation()
        peer = PeerState(msg_count=100, T_peer=0.1, count_below=0)
        self.assertGreater(c.revocation_hint(peer), 0.0)

    def test_cadence_is_constant(self) -> None:
        """No cadence adaptation in detection-only baseline."""
        c = NaiveThresholdRevocation()
        for kappa in (0.0, 5.0, 50.0):
            nbh = NeighbourhoodSummary(
                mean=0.4, q25=0.3, n=10, rho_t=15.0, kappa_t=kappa
            )
            self.assertEqual(c.cadence(nbh), c.cfg.c_base)

    def test_registered_in_coupling_variants(self) -> None:
        self.assertIn("naive_revocation", COUPLING_VARIANTS)
        self.assertIs(
            COUPLING_VARIANTS["naive_revocation"], NaiveThresholdRevocation
        )


if __name__ == "__main__":
    unittest.main()
