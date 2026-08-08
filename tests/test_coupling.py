"""Verifies coupling layer against properties P1 to P10 from coupling_spec.md.

Run from `backend/` directory:
    python -m unittest discover -s tests -v
"""
import unittest

from piad_v2x.lifecycle.coupling import (
    CouplingConfig,
    CouplingLayer,
    NeighbourhoodSummary,
    PeerState,
)


class CadenceProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CouplingLayer()
        self.cfg = self.layer.cfg

    def test_p1_healthy_neighbourhood_uses_base_cadence(self) -> None:
        """P1: N_t.mean >= theta_safe => C_self = C_base."""
        n = NeighbourhoodSummary(mean=0.9, q25=0.85, n=10)
        self.assertAlmostEqual(self.layer.cadence(n), self.cfg.C_base)

    def test_p2_cadence_bounded_above_by_C_max(self) -> None:
        """P2: C_self <= C_max for all inputs."""
        n = NeighbourhoodSummary(mean=0.0, q25=0.0, n=20)
        self.assertLessEqual(self.layer.cadence(n), self.cfg.C_max)

    def test_p3_cadence_bounded_below_by_C_min(self) -> None:
        """P3: C_self >= C_min for all inputs."""
        for mean in [0.0, 0.5, 1.0]:
            n = NeighbourhoodSummary(mean=mean, q25=mean, n=20)
            self.assertGreaterEqual(self.layer.cadence(n), self.cfg.C_min)

    def test_p4_cadence_continuous_in_mean(self) -> None:
        """P4: small change in N_t.mean => small change in C_self."""
        a = NeighbourhoodSummary(mean=0.5, q25=0.4, n=10)
        b = NeighbourhoodSummary(mean=0.501, q25=0.4, n=10)
        diff = abs(self.layer.cadence(a) - self.layer.cadence(b))
        self.assertLess(diff, 1e-3)

    def test_p5_reliability_gating_falls_back_to_base(self) -> None:
        """P5: small neighbourhoods (n < n_min) => C_self = C_base."""
        n = NeighbourhoodSummary(mean=0.0, q25=0.0, n=2)  # below n_min=3
        self.assertAlmostEqual(self.layer.cadence(n), self.cfg.C_base)

    def test_cadence_grows_when_neighbourhood_unhealthy(self) -> None:
        """Sanity: lower mean trust => higher cadence (within bounds)."""
        healthy = NeighbourhoodSummary(mean=0.8, q25=0.7, n=10)
        unhealthy = NeighbourhoodSummary(mean=0.4, q25=0.3, n=10)
        self.assertGreater(self.layer.cadence(unhealthy), self.layer.cadence(healthy))


class RevocationProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = CouplingLayer()
        self.cfg = self.layer.cfg

    def test_p6_revocation_in_unit_interval(self) -> None:
        """P6: R_p in [0, 1]."""
        for T in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
            peer = PeerState(msg_count=100, T_peer=T, count_below=50)
            r = self.layer.revocation_hint(peer)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)

    def test_p7_cold_start_zero(self) -> None:
        """P7: R_p = 0 in first w/2 messages even with very low trust."""
        peer = PeerState(msg_count=10, T_peer=0.0, count_below=10)
        self.assertEqual(self.layer.revocation_hint(peer), 0.0)

    def test_p8_above_revoke_threshold_zero(self) -> None:
        """P8: T_peer >= theta_revoke => R_p = 0."""
        peer = PeerState(msg_count=100, T_peer=0.4, count_below=50)
        self.assertEqual(self.layer.revocation_hint(peer), 0.0)

    def test_persistence_required_below_k(self) -> None:
        """count_below < k => R_p = 0 even with very low T_peer."""
        peer = PeerState(msg_count=100, T_peer=0.0, count_below=5)
        self.assertEqual(self.layer.revocation_hint(peer), 0.0)

    def test_p9_monotone_non_increasing_in_T_peer(self) -> None:
        """P9: R_p non-increasing in T_peer within active region."""
        a = PeerState(msg_count=100, T_peer=0.05, count_below=50)
        b = PeerState(msg_count=100, T_peer=0.15, count_below=50)
        c = PeerState(msg_count=100, T_peer=0.25, count_below=50)
        self.assertGreaterEqual(self.layer.revocation_hint(a), self.layer.revocation_hint(b))
        self.assertGreaterEqual(self.layer.revocation_hint(b), self.layer.revocation_hint(c))

    def test_revocation_hints_dict_passes_through(self) -> None:
        peers = {
            "good": PeerState(msg_count=100, T_peer=0.9, count_below=0),
            "bad": PeerState(msg_count=100, T_peer=0.05, count_below=50),
        }
        hints = self.layer.revocation_hints(peers)
        self.assertEqual(hints["good"], 0.0)
        self.assertGreater(hints["bad"], 0.0)


class DefenceClaims(unittest.TestCase):
    """Sanity checks on threat_model.md defence claims D1, D2, D3."""

    def test_d1_persistent_attacker_triggers_revocation_after_persistence(self) -> None:
        """D1: persistent low-trust attacker -> R_p > 0 once persistence and cold-start are passed."""
        layer = CouplingLayer()
        peer = PeerState(msg_count=30, T_peer=0.05, count_below=25)
        self.assertGreater(layer.revocation_hint(peer), 0.0)

    def test_d2_rotation_churn_bounded_by_C_max(self) -> None:
        """D2: cadence cannot exceed C_max regardless of adversarial inputs.

        Note: with default config the operating ceiling is C_base * (1 + alpha * theta_safe)
        = (1/60) * 4.5 = 0.075, which is below C_max = 0.1. C_max is a hard safety bound
        rather than the typical operating point. The claim is C_self <= C_max, not equality.
        """
        layer = CouplingLayer()
        worst = NeighbourhoodSummary(mean=0.0, q25=0.0, n=1000)
        self.assertLessEqual(layer.cadence(worst), layer.cfg.C_max)

    def test_d2_C_max_clamp_engages_under_aggressive_alpha(self) -> None:
        """D2 (clamp engagement): with alpha large enough, the C_max clamp does cap output."""
        cfg = CouplingConfig(alpha=20.0)  # 1 + 20*0.7 = 15 -> way above C_max/C_base
        layer = CouplingLayer(cfg)
        worst = NeighbourhoodSummary(mean=0.0, q25=0.0, n=10)
        self.assertAlmostEqual(layer.cadence(worst), cfg.C_max)

    def test_d3_cold_start_safety(self) -> None:
        """D3: no revocation in first w/2 messages, regardless of detector verdict."""
        layer = CouplingLayer()
        for msg_count in range(0, layer.cfg.cold_start_msgs):
            peer = PeerState(msg_count=msg_count, T_peer=0.0, count_below=msg_count)
            self.assertEqual(
                layer.revocation_hint(peer),
                0.0,
                msg=f"R_p should be 0 at msg_count={msg_count}",
            )


class PhysicsInformedModulators(unittest.TestCase):
    """DD-007: g_density(ρ_t) and g_churn(κ_t) shape the cadence response."""

    def setUp(self) -> None:
        self.layer = CouplingLayer()
        self.cfg = self.layer.cfg

    def _hostile_summary(self, *, rho_t: float, kappa_t: float) -> NeighbourhoodSummary:
        return NeighbourhoodSummary(
            mean=0.2, q25=0.1, n=20, rho_t=rho_t, drho_dt=0.0, kappa_t=kappa_t,
        )

    def test_g_density_no_amplification_at_or_below_rho_low(self) -> None:
        """ρ_t <= ρ_low => g_density = 1 (no amplification)."""
        at_low = self.layer.cadence(self._hostile_summary(rho_t=self.cfg.rho_low, kappa_t=0.0))
        below_low = self.layer.cadence(self._hostile_summary(rho_t=0.0, kappa_t=0.0))
        self.assertAlmostEqual(at_low, below_low)

    def test_g_density_full_amplification_at_or_above_rho_high(self) -> None:
        """ρ_t >= ρ_high => g_density saturates at 1 + γ_ρ."""
        at_high = self.layer.cadence(self._hostile_summary(rho_t=self.cfg.rho_high, kappa_t=0.0))
        way_above = self.layer.cadence(self._hostile_summary(rho_t=self.cfg.rho_high * 5, kappa_t=0.0))
        self.assertAlmostEqual(at_high, way_above)

    def test_g_density_monotone_in_rho_t(self) -> None:
        """In the active band, more density => more cadence."""
        cadences = [
            self.layer.cadence(self._hostile_summary(rho_t=r, kappa_t=0.0))
            for r in [self.cfg.rho_low, 15.0, self.cfg.rho_high]
        ]
        self.assertLess(cadences[0], cadences[1])
        self.assertLess(cadences[1], cadences[2])

    def test_g_churn_no_damping_at_or_below_kappa_safe(self) -> None:
        """κ_t <= κ_safe => g_churn = 1 (no damping)."""
        at_safe = self.layer.cadence(self._hostile_summary(rho_t=20.0, kappa_t=self.cfg.kappa_safe))
        below_safe = self.layer.cadence(self._hostile_summary(rho_t=20.0, kappa_t=0.0))
        self.assertAlmostEqual(at_safe, below_safe)

    def test_g_churn_damps_under_high_churn(self) -> None:
        """High κ_t damps the cadence response (prevents rotation-DoS amplification)."""
        low_churn = self.layer.cadence(self._hostile_summary(rho_t=20.0, kappa_t=0.0))
        high_churn = self.layer.cadence(
            self._hostile_summary(rho_t=20.0, kappa_t=self.cfg.kappa_safe * 5),
        )
        self.assertLess(high_churn, low_churn)

    def test_g_churn_saturates_at_high_churn(self) -> None:
        """κ_t >> κ_safe => damping factor saturates at 1/(1+γ_κ)."""
        very_high = self.layer.cadence(self._hostile_summary(rho_t=20.0, kappa_t=100.0))
        extreme = self.layer.cadence(self._hostile_summary(rho_t=20.0, kappa_t=1.0e6))
        self.assertAlmostEqual(very_high, extreme)

    def test_dd007_hostile_dense_low_churn_vs_high_churn(self) -> None:
        """Operational claim: under identical trust+density, high churn
        produces lower cadence than low churn. This is what stops a Sybil swarm
        from weaponising its own churn signal."""
        dense_low_churn = self.layer.cadence(self._hostile_summary(rho_t=25.0, kappa_t=0.0))
        dense_high_churn = self.layer.cadence(self._hostile_summary(rho_t=25.0, kappa_t=5.0))
        self.assertGreater(dense_low_churn, dense_high_churn)

    def test_dd007_density_modulator_still_bounded_by_C_max(self) -> None:
        """g_density amplification cannot breach C_max."""
        worst = NeighbourhoodSummary(
            mean=0.0, q25=0.0, n=100, rho_t=200.0, drho_dt=0.0, kappa_t=0.0,
        )
        self.assertLessEqual(self.layer.cadence(worst), self.cfg.C_max)


class PhysicsConditionedRevocationGate(unittest.TestCase):
    """DD-008 S4: θ_revoke is conditioned on neighbourhood physics (ρ_t, κ_t).

    Verifies properties P11 to P14 of coupling_spec.md §4.4.
    """

    def setUp(self) -> None:
        # S4 is opt-in (gamma_revoke defaults to 0.0); enable it explicitly.
        self.layer = CouplingLayer(CouplingConfig(gamma_revoke=0.5))
        self.cfg = self.layer.cfg

    def _ctx(self, *, rho_t: float, kappa_t: float) -> NeighbourhoodSummary:
        return NeighbourhoodSummary(
            mean=0.5, q25=0.4, n=20, rho_t=rho_t, drho_dt=0.0, kappa_t=kappa_t,
        )

    def test_s4_off_by_default_so_wiring_is_a_noop(self) -> None:
        """gamma_revoke defaults to 0.0: density does NOT change the gate, so
        passing the neighbourhood through the simulator is inert until S4 is
        explicitly enabled. This is what makes the integration safe."""
        default_layer = CouplingLayer()  # gamma_revoke = 0.0
        self.assertEqual(default_layer.cfg.gamma_revoke, 0.0)
        peer = PeerState(msg_count=100, T_peer=0.44, count_below=50)
        dense = self._ctx(rho_t=self.cfg.rho_high, kappa_t=0.0)
        self.assertAlmostEqual(
            default_layer.effective_theta_revoke(dense), default_layer.cfg.theta_revoke
        )
        self.assertEqual(
            default_layer.revocation_hint(peer, dense),
            default_layer.revocation_hint(peer),
        )

    def test_p14_neutral_context_matches_fixed_gate(self) -> None:
        """P14: no summary, or ρ_t <= ρ_low, gives the pre-S4 fixed threshold."""
        peer = PeerState(msg_count=100, T_peer=0.25, count_below=50)
        fixed = self.layer.revocation_hint(peer)  # no neighbourhood
        neutral = self.layer.revocation_hint(peer, self._ctx(rho_t=0.0, kappa_t=0.0))
        at_low = self.layer.revocation_hint(peer, self._ctx(rho_t=self.cfg.rho_low, kappa_t=0.0))
        self.assertEqual(fixed, neutral)
        self.assertEqual(fixed, at_low)
        self.assertAlmostEqual(self.layer.effective_theta_revoke(None), self.cfg.theta_revoke)

    def test_p11_effective_threshold_bounded_below_theta_low(self) -> None:
        """P11: θ_revoke^eff stays in [θ_revoke, θ_revoke_max] ⊂ [_, 0.5)."""
        for rho in [0.0, 5.0, 15.0, 30.0, 1000.0]:
            for kap in [0.0, 1.0, 50.0]:
                eff = self.layer.effective_theta_revoke(self._ctx(rho_t=rho, kappa_t=kap))
                self.assertGreaterEqual(eff, self.cfg.theta_revoke - 1e-12)
                self.assertLessEqual(eff, self.cfg.theta_revoke_max + 1e-12)
                self.assertLess(eff, 0.5)
        # A peer just above the ceiling is never flagged, even at extreme context.
        peer = PeerState(msg_count=100, T_peer=0.46, count_below=50)
        self.assertEqual(
            self.layer.revocation_hint(peer, self._ctx(rho_t=1.0e6, kappa_t=0.0)), 0.0
        )

    def test_p11_base_above_cap_is_inert(self) -> None:
        """If the configured theta_revoke already exceeds theta_revoke_max,
        density must NOT lower it (regression guard: the grid sweeps
        theta_revoke in {0.30, 0.50, 0.70}, and 0.50/0.70 exceed the 0.45 cap)."""
        layer = CouplingLayer(CouplingConfig(theta_revoke=0.50, gamma_revoke=0.5))
        for rho in [0.0, 15.0, layer.cfg.rho_high, 1.0e6]:
            eff = layer.effective_theta_revoke(self._ctx(rho_t=rho, kappa_t=0.0))
            self.assertAlmostEqual(eff, 0.50)  # unchanged: no room to rise

    def test_density_flips_a_borderline_peer_into_the_gate(self) -> None:
        """The core S4 behaviour: a persistently-low peer that the fixed gate
        leaves alone (T_peer above 0.3) is flagged once organic density raises
        θ_revoke^eff above its trust."""
        peer = PeerState(msg_count=100, T_peer=0.44, count_below=50)
        self.assertEqual(self.layer.revocation_hint(peer), 0.0)  # fixed gate: 0.44 >= 0.3
        dense = self.layer.revocation_hint(peer, self._ctx(rho_t=self.cfg.rho_high, kappa_t=0.0))
        self.assertGreater(dense, 0.0)  # θ_revoke^eff = 0.45 > 0.44

    def test_p12_monotone_non_decreasing_in_density(self) -> None:
        """P12: more density => θ_revoke^eff up => R_p non-decreasing."""
        peer = PeerState(msg_count=100, T_peer=0.40, count_below=50)
        rs = [
            self.layer.revocation_hint(peer, self._ctx(rho_t=r, kappa_t=0.0))
            for r in [0.0, 15.0, self.cfg.rho_high]
        ]
        self.assertLessEqual(rs[0], rs[1])
        self.assertLessEqual(rs[1], rs[2])
        self.assertGreater(rs[2], 0.0)

    def test_p13_churn_damps_the_gate_anti_weaponisation(self) -> None:
        """P13: under identical density, high churn (Sybil-swarm signature)
        damps θ_revoke^eff, so the swarm cannot inflate the gate against a
        borderline benign peer."""
        peer = PeerState(msg_count=100, T_peer=0.42, count_below=50)
        low_churn = self.layer.revocation_hint(peer, self._ctx(rho_t=self.cfg.rho_high, kappa_t=0.0))
        high_churn = self.layer.revocation_hint(peer, self._ctx(rho_t=self.cfg.rho_high, kappa_t=5.0))
        self.assertGreater(low_churn, 0.0)        # dense + calm: gate is up, peer flagged
        self.assertLess(high_churn, low_churn)    # dense + churny: damped back
        # effective threshold itself is non-increasing in churn
        eff_low = self.layer.effective_theta_revoke(self._ctx(rho_t=self.cfg.rho_high, kappa_t=0.0))
        eff_high = self.layer.effective_theta_revoke(self._ctx(rho_t=self.cfg.rho_high, kappa_t=5.0))
        self.assertLess(eff_high, eff_low)

    def test_revocation_hints_dict_passes_context(self) -> None:
        peers = {"borderline": PeerState(msg_count=100, T_peer=0.44, count_below=50)}
        fixed = self.layer.revocation_hints(peers)
        dense = self.layer.revocation_hints(peers, self._ctx(rho_t=self.cfg.rho_high, kappa_t=0.0))
        self.assertEqual(fixed["borderline"], 0.0)
        self.assertGreater(dense["borderline"], 0.0)


class CustomConfig(unittest.TestCase):
    def test_custom_config_respected(self) -> None:
        cfg = CouplingConfig(
            C_base=1.0,
            C_min=0.1,
            C_max=10.0,
            alpha=2.0,
            theta_safe=0.5,
            n_min=1,
            theta_revoke=0.4,
            k=3,
            beta=1.0,
            cold_start_msgs=5,
        )
        layer = CouplingLayer(cfg)
        n = NeighbourhoodSummary(mean=0.6, q25=0.5, n=2)  # above theta_safe
        self.assertAlmostEqual(layer.cadence(n), 1.0)
        n2 = NeighbourhoodSummary(mean=0.0, q25=0.0, n=2)
        # f = 1 + 2*0.5 = 2; C_base*f = 2; below C_max=10
        self.assertAlmostEqual(layer.cadence(n2), 2.0)


if __name__ == "__main__":
    unittest.main()
