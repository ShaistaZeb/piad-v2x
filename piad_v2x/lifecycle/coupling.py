"""Coupling layer.

Implements the formal cadence and revocation mapping:
- C_self(t): rotation cadence directive for the vehicle's own pseudonym.
- R_p(t):    per-peer revocation hint score.

Pure-Python; no external dependencies. The mapping is recomputed from
the trust aggregator state on every call. There is no internal state machine.

Two physics-informed modulators shape the cadence formula:
- g_density(ρ_t): high local vehicle density amplifies the trust response.
- g_churn(κ_t):   high pseudonym-churn rate damps the response, so a
                  rotation-DoS adversary cannot weaponise its own churn.

The same physics-conditioning extends to the revocation gate:
the effective revocation threshold rises with organic local density (more
attack surface) but is damped by churn (the rotation-DoS signature), so the
gate cannot be weaponised by an adversary inflating its own churn.

The trust aggregator is responsible for measuring ρ_t, ∂ρ/∂t, and κ_t
and surfacing them through `NeighbourhoodSummary`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class CouplingConfig:
    # Cadence parameters
    C_base: float = 1.0 / 60.0
    C_min: float = 1.0 / 300.0
    C_max: float = 1.0 / 10.0
    alpha: float = 5.0
    theta_safe: float = 0.7
    n_min: int = 3

    # Physics-informed modulators
    gamma_rho: float = 0.5
    rho_low: float = 5.0
    rho_high: float = 30.0
    gamma_kappa: float = 0.6
    kappa_safe: float = 1.0

    # Revocation parameters
    theta_revoke: float = 0.3
    k: int = 20
    beta: float = 2.0
    cold_start_msgs: int = 25

    # Physics-conditioned revocation gate.
    # The effective theta_revoke rises with organic local density, damped by
    # churn so a rotation-DoS swarm cannot inflate the gate. theta_revoke_max
    # is held below theta_low = 0.5 so the gate never fires on mere uncertainty.
    #
    # gamma_revoke defaults to 0.0: the gate is INERT by default, so wiring it into
    # the simulator changes no existing (gate-off) experiment. Set it to the
    # evaluation strength (0.5) explicitly to enable the gate. It stays opt-in
    # until the revocation-frontier diagnostic (prereg E.2) validates it.
    gamma_revoke: float = 0.0
    theta_revoke_max: float = 0.45


@dataclass(frozen=True)
class NeighbourhoodSummary:
    """Trust + physics-informed summary of the local neighbourhood at time t.

    Earlier callers that only supply (mean, q25, n) still work; the physics
    fields default to neutral values that disable the new modulators.
    """
    mean: float
    q25: float
    n: int
    rho_t: float = 0.0
    drho_dt: float = 0.0
    kappa_t: float = 0.0


@dataclass(frozen=True)
class PeerState:
    msg_count: int
    T_peer: float
    count_below: int


def _clamp(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


class CouplingLayer:
    def __init__(self, config: CouplingConfig | None = None) -> None:
        self.cfg = config or CouplingConfig()

    def cadence(self, neighbourhood: NeighbourhoodSummary) -> float:
        """C_self(t). Rotation cadence.

        C_self(t) = clamp(C_min, C_max,
                          C_base * (1 + alpha * max(0, theta_safe - mean)
                                          * g_density(rho_t) * g_churn(kappa_t)))
        Reliability gate: if n < n_min, fall back to C_base.
        """
        cfg = self.cfg
        if neighbourhood.n < cfg.n_min:
            return cfg.C_base
        trust_term = max(0.0, cfg.theta_safe - neighbourhood.mean)
        f = 1.0 + cfg.alpha * trust_term \
            * self._g_density(neighbourhood.rho_t) \
            * self._g_churn(neighbourhood.kappa_t)
        return _clamp(cfg.C_min, cfg.C_max, cfg.C_base * f)

    def _g_density(self, rho_t: float) -> float:
        """g_density(ρ_t) = 1 + γ_ρ · clamp(0, 1, (ρ_t - ρ_low) / (ρ_high - ρ_low)).

        Amplifies the trust response under high local density: more peers
        means more attack surface, so faster rotation is warranted.
        """
        cfg = self.cfg
        if cfg.rho_high <= cfg.rho_low:
            return 1.0
        normalised = (rho_t - cfg.rho_low) / (cfg.rho_high - cfg.rho_low)
        return 1.0 + cfg.gamma_rho * _clamp(0.0, 1.0, normalised)

    def _g_churn(self, kappa_t: float) -> float:
        """g_churn(κ_t) = 1 / (1 + γ_κ · clamp(0, 1, (κ_t - κ_safe) / κ_safe)).

        Damps the response when pseudonym-churn rate exceeds the safe baseline.
        Prevents a coordinated rotation-DoS adversary from amplifying its own
        churn signal through our cadence response.
        """
        cfg = self.cfg
        if cfg.kappa_safe <= 0.0:
            return 1.0
        normalised = (kappa_t - cfg.kappa_safe) / cfg.kappa_safe
        return 1.0 / (1.0 + cfg.gamma_kappa * _clamp(0.0, 1.0, normalised))

    def _revoke_context_gain(self, rho_t: float, kappa_t: float) -> float:
        """a_ctx(ρ_t, κ_t) ∈ [0, 1]: the physics context gain.

        a_ctx = density_norm(ρ_t) · g_churn(κ_t)

        High only when the neighbourhood is dense AND low-churn (organic
        density). g_churn (the same factor used by the cadence) damps the gain
        when churn is high, so a rotation-DoS Sybil swarm that inflates ρ_t by
        rapid rotation cannot drive the revocation gate up against benign peers.
        """
        cfg = self.cfg
        if cfg.rho_high <= cfg.rho_low:
            density_norm = 0.0
        else:
            density_norm = _clamp(
                0.0, 1.0, (rho_t - cfg.rho_low) / (cfg.rho_high - cfg.rho_low)
            )
        return density_norm * self._g_churn(kappa_t)

    def effective_theta_revoke(
        self, neighbourhood: NeighbourhoodSummary | None = None
    ) -> float:
        """θ_revoke^eff(t) ∈ [θ_revoke, θ_revoke_max].

        With no neighbourhood summary (or neutral context ρ_t ≤ ρ_low) this is
        exactly θ_revoke, so the gate reduces to the fixed threshold.
        """
        cfg = self.cfg
        if neighbourhood is None:
            return cfg.theta_revoke
        a_ctx = self._revoke_context_gain(neighbourhood.rho_t, neighbourhood.kappa_t)
        raised = cfg.theta_revoke * (1.0 + cfg.gamma_revoke * a_ctx)
        # Density can only RAISE the threshold toward theta_revoke_max, never
        # lower it below the configured base. If the base already meets or
        # exceeds theta_revoke_max, there is no room to rise and the gate is inert.
        return max(cfg.theta_revoke, min(cfg.theta_revoke_max, raised))

    def revocation_hint(
        self,
        peer: PeerState,
        neighbourhood: NeighbourhoodSummary | None = None,
    ) -> float:
        """R_p(t). Per-peer revocation hint.

        When `neighbourhood` is supplied the revocation threshold is physics-
        conditioned: organic density raises θ_revoke toward
        θ_revoke_max, damped by churn. Without it, behaviour is identical to the
        fixed-threshold gate.
        """
        cfg = self.cfg
        theta_rev = self.effective_theta_revoke(neighbourhood)
        if peer.msg_count < cfg.cold_start_msgs:
            return 0.0
        if peer.T_peer >= theta_rev:
            return 0.0
        if peer.count_below < cfg.k:
            return 0.0
        diff = theta_rev - peer.T_peer
        return _clamp(0.0, 1.0, diff ** cfg.beta)

    def revocation_hints(
        self,
        peers: Mapping[str, PeerState],
        neighbourhood: NeighbourhoodSummary | None = None,
    ) -> Dict[str, float]:
        return {p: self.revocation_hint(state, neighbourhood) for p, state in peers.items()}
