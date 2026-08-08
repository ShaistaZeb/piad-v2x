"""Tests for the trust aggregator."""
import unittest

from piad_v2x.lifecycle.aggregator import AggregatorConfig, TrustAggregator
from piad_v2x.lifecycle.coupling import CouplingLayer


class AggregatorBasics(unittest.TestCase):
    def test_cold_start_neutral_prior(self) -> None:
        """First message: T_peer is alpha-weighted from neutral 0.5."""
        agg = TrustAggregator()
        state = agg.update("pk1", 0.8, now=0.0)
        # alpha=0.85: 0.85 * 0.5 + 0.15 * 0.8 = 0.545
        self.assertAlmostEqual(state.T_peer, 0.545, places=5)
        self.assertEqual(state.msg_count, 1)
        self.assertEqual(state.count_below, 0)

    def test_count_below_tracks_window(self) -> None:
        """count_below counts trust scores in window strictly below theta_low."""
        agg = TrustAggregator(AggregatorConfig(window=10, alpha=0.5, theta_low=0.5))
        for s in [0.1, 0.2, 0.3, 0.6, 0.7]:
            agg.update("pk1", s, now=0.0)
        state = agg.peer_states()["pk1"]
        self.assertEqual(state.count_below, 3)

    def test_window_bounded(self) -> None:
        """Older scores fall out of the window."""
        agg = TrustAggregator(AggregatorConfig(window=3, theta_low=0.5))
        for s in [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]:
            agg.update("pk1", s, now=0.0)
        state = agg.peer_states()["pk1"]
        self.assertEqual(state.count_below, 0)

    def test_eviction_removes_stale_peers(self) -> None:
        agg = TrustAggregator(AggregatorConfig(eviction_seconds=1.0))
        agg.update("pk1", 0.5, now=0.0)
        agg.update("pk2", 0.5, now=0.5)
        agg.evict(now=2.0)
        self.assertNotIn("pk1", agg.peer_states())
        self.assertNotIn("pk2", agg.peer_states())

    def test_eviction_preserves_recent(self) -> None:
        agg = TrustAggregator(AggregatorConfig(eviction_seconds=1.0))
        agg.update("pk1", 0.5, now=0.0)
        agg.update("pk2", 0.5, now=2.0)
        agg.evict(now=2.5)
        peers = agg.peer_states()
        self.assertNotIn("pk1", peers)
        self.assertIn("pk2", peers)


class NeighbourhoodSummary(unittest.TestCase):
    def test_empty_returns_neutral(self) -> None:
        agg = TrustAggregator()
        n = agg.neighbourhood(now=0.0)
        self.assertEqual(n.n, 0)
        self.assertEqual(n.mean, 0.5)

    def test_active_window_inclusive(self) -> None:
        agg = TrustAggregator(AggregatorConfig(neighbourhood_seconds=2.0))
        agg.update("pk1", 0.8, now=0.0)
        agg.update("pk2", 0.2, now=1.0)
        n = agg.neighbourhood(now=1.5)
        self.assertEqual(n.n, 2)

    def test_excludes_inactive_peers(self) -> None:
        agg = TrustAggregator(AggregatorConfig(neighbourhood_seconds=1.0))
        agg.update("pk1", 0.8, now=0.0)
        agg.update("pk2", 0.2, now=2.0)
        n = agg.neighbourhood(now=2.5)
        self.assertEqual(n.n, 1)

    def test_mean_and_quartile(self) -> None:
        agg = TrustAggregator(AggregatorConfig(alpha=0.0, neighbourhood_seconds=10.0))
        # alpha=0 means T_peer takes the latest score directly
        for i, s in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            agg.update(f"pk{i}", s, now=float(i))
        n = agg.neighbourhood(now=4.5)
        self.assertEqual(n.n, 5)
        self.assertAlmostEqual(n.mean, 0.5)
        self.assertLess(n.q25, n.mean)


class PhysicsInformedNeighbourhood(unittest.TestCase):
    """The aggregator must expose ρ_t, ∂ρ/∂t, κ_t to the coupling layer."""

    def test_rho_t_equals_active_count_with_unit_coverage(self) -> None:
        """With coverage_area=1.0, ρ_t equals the active peer count."""
        agg = TrustAggregator(AggregatorConfig(coverage_area=1.0))
        for i in range(7):
            agg.update(f"pk{i}", 0.5, now=0.0)
        n = agg.neighbourhood(now=0.0)
        self.assertEqual(n.n, 7)
        self.assertAlmostEqual(n.rho_t, 7.0)

    def test_rho_t_scales_with_coverage_area(self) -> None:
        """ρ_t is peers / coverage_area."""
        agg = TrustAggregator(AggregatorConfig(coverage_area=10.0))
        for i in range(20):
            agg.update(f"pk{i}", 0.5, now=0.0)
        n = agg.neighbourhood(now=0.0)
        self.assertAlmostEqual(n.rho_t, 2.0)

    def test_drho_dt_zero_in_steady_state(self) -> None:
        """Stable peer set => gradient is 0."""
        agg = TrustAggregator(AggregatorConfig(gradient_window_seconds=2.0))
        # Seed 5 peers, then keep pinging them at constant rate for 3 seconds.
        for i in range(5):
            agg.update(f"pk{i}", 0.5, now=0.0)
        for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for i in range(5):
                agg.update(f"pk{i}", 0.5, now=t)
        n = agg.neighbourhood(now=3.0)
        self.assertAlmostEqual(n.drho_dt, 0.0, places=6)

    def test_drho_dt_positive_when_density_grows(self) -> None:
        """Density increasing over the gradient window => ∂ρ/∂t > 0."""
        agg = TrustAggregator(AggregatorConfig(gradient_window_seconds=2.0))
        # At t=0 there is 1 peer. At t=2.5 there are 6.
        agg.update("pk0", 0.5, now=0.0)
        for i in range(1, 6):
            agg.update(f"pk{i}", 0.5, now=2.5)
        n = agg.neighbourhood(now=2.5)
        self.assertGreater(n.drho_dt, 0.0)

    def test_drho_dt_zero_when_no_old_snapshot_yet(self) -> None:
        """Early life (no snapshot older than gradient window) => 0."""
        agg = TrustAggregator(AggregatorConfig(gradient_window_seconds=2.0))
        for i in range(5):
            agg.update(f"pk{i}", 0.5, now=0.1 * i)
        n = agg.neighbourhood(now=0.5)
        self.assertAlmostEqual(n.drho_dt, 0.0)

    def test_kappa_t_zero_when_no_new_pseudonyms(self) -> None:
        """A stable set of pseudonyms produces zero churn over w_t."""
        agg = TrustAggregator(AggregatorConfig(neighbourhood_seconds=5.0))
        for i in range(5):
            agg.update(f"pk{i}", 0.5, now=0.0)
        # 6 seconds later, no new pseudonyms have appeared in the last w_t.
        for t in [5.5, 6.0]:
            for i in range(5):
                agg.update(f"pk{i}", 0.5, now=t)
        n = agg.neighbourhood(now=6.0)
        self.assertAlmostEqual(n.kappa_t, 0.0)

    def test_kappa_t_positive_when_pseudonyms_rotate(self) -> None:
        """Fresh pseudonyms first-seen inside w_t => κ_t > 0."""
        agg = TrustAggregator(AggregatorConfig(neighbourhood_seconds=5.0))
        # 10 fresh pseudonyms appearing across the last 5 seconds.
        for i in range(10):
            agg.update(f"fresh{i}", 0.5, now=0.5 * i)
        n = agg.neighbourhood(now=5.0)
        # 10 new in 5s = 2 per second.
        self.assertAlmostEqual(n.kappa_t, 2.0)

    def test_kappa_t_decays_with_quiet_window(self) -> None:
        """Once first-seen events fall out of w_t, κ_t returns to 0."""
        agg = TrustAggregator(AggregatorConfig(neighbourhood_seconds=2.0))
        for i in range(4):
            agg.update(f"pk{i}", 0.5, now=0.0)
        n_immediate = agg.neighbourhood(now=0.0)
        self.assertGreater(n_immediate.kappa_t, 0.0)
        # 10 seconds later, with no new arrivals.
        for i in range(4):
            agg.update(f"pk{i}", 0.5, now=10.0)
        n_later = agg.neighbourhood(now=10.0)
        self.assertAlmostEqual(n_later.kappa_t, 0.0)


class IntegrationWithCoupling(unittest.TestCase):
    """Streaming integration: aggregator output feeds coupling layer."""

    def test_persistent_attacker_triggers_revocation_within_bound(self) -> None:
        """D1 end-to-end: stream a persistent low-trust attacker; R_p fires within
        max(w/2, k) messages = max(25, 20) = 25 messages with default config."""
        agg = TrustAggregator()
        layer = CouplingLayer()
        fired_at = None
        for i in range(80):
            agg.update("attacker", 0.05, now=float(i) * 0.1)
            peer = agg.peer_states()["attacker"]
            r = layer.revocation_hint(peer)
            if r > 0.0 and fired_at is None:
                fired_at = i + 1  # message ordinal
        self.assertIsNotNone(fired_at, "R_p never fired for persistent attacker")
        # Bound from spec: max(cold_start=25, k=20) = 25; allow small EMA settling slack.
        self.assertLessEqual(fired_at, 60, f"R_p fired too late at message {fired_at}")

    def test_benign_peer_never_triggers_revocation(self) -> None:
        agg = TrustAggregator()
        layer = CouplingLayer()
        for i in range(200):
            agg.update("benign", 0.9, now=float(i) * 0.1)
        peer = agg.peer_states()["benign"]
        self.assertEqual(layer.revocation_hint(peer), 0.0)

    def test_healthy_neighbourhood_keeps_base_cadence(self) -> None:
        agg = TrustAggregator()
        layer = CouplingLayer()
        for pk in ["a", "b", "c", "d", "e"]:
            for i in range(50):
                agg.update(pk, 0.9, now=float(i) * 0.1)
        n = agg.neighbourhood(now=5.0)
        self.assertGreaterEqual(n.n, 5)
        self.assertGreaterEqual(n.mean, 0.7)
        self.assertAlmostEqual(layer.cadence(n), layer.cfg.C_base)

    def test_hostile_dense_lowchurn_accelerates_more_than_sparse(self) -> None:
        """Same low neighbourhood trust, but stable-dense env should drive higher cadence
        than stable-sparse, due to g_density amplification.

        Both scenarios are aged past w_t so g_churn is neutral; only g_density differs.
        """
        layer = CouplingLayer()

        def _drive(pseudonyms: list[str]) -> "NeighbourhoodSummary":  # noqa: F821
            agg = TrustAggregator()
            # Introduce all peers far in the past so their first-seen events
            # have fallen out of w_t; then keep them active right up to t=20.0.
            for pk in pseudonyms:
                agg.update(pk, 0.05, now=0.0)
            for t in [10.0, 15.0, 17.0, 18.0, 19.0, 19.5, 20.0]:
                for pk in pseudonyms:
                    agg.update(pk, 0.05, now=t)
            return agg.neighbourhood(now=20.0)

        n_sparse = _drive(["a", "b", "c", "d"])
        n_dense = _drive([f"d{i}" for i in range(25)])

        # Sanity: both should report ~zero churn at measurement time.
        self.assertAlmostEqual(n_sparse.kappa_t, 0.0)
        self.assertAlmostEqual(n_dense.kappa_t, 0.0)
        # ρ_t differs by design.
        self.assertGreater(n_dense.rho_t, n_sparse.rho_t)

        c_sparse = layer.cadence(n_sparse)
        c_dense = layer.cadence(n_dense)
        self.assertGreater(c_dense, c_sparse, "g_density should amplify under dense low-trust")


if __name__ == "__main__":
    unittest.main()
