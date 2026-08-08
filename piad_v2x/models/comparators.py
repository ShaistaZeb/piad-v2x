"""Coupling-layer comparators for the three-baseline study.

These are decoupled variants of the full coupling layer, each of which
exposes the same `cadence(neighbourhood) -> float` and
`revocation_hint(peer_state, neighbourhood=None) -> float` interface as
`CouplingLayer`, so they can be slotted into the streaming simulator
interchangeably. The baselines accept `neighbourhood` for signature parity but
ignore it; only `CouplingLayer` uses it to physics-condition the gate.

The three comparators map onto the comparator papers cited in the literature
audit and in the dissertation prep materials:

- ContextOnlyCoupling   (SAPACS-style): rotation cadence reacts to density
                                        and configured privacy level only;
                                        no detector input. Cannot distinguish
                                        benign-dense from hostile-dense.
- TrustOnlyCoupling     (MP-TMD-style): rotation cadence reacts to per-peer
                                        trust only; no physics modulators
                                        (no g_density, no g_churn). The earlier
                                        trust-only form of the framework.
- NoCoupling            (baseline):     fixed cadence at C_base, no
                                        revocation hints. The "do nothing"
                                        floor for comparison.

The hint surface is honoured by every variant for fairness in the
end-to-end pipeline: NoCoupling returns 0 always; ContextOnly returns 0
always (no detector input); TrustOnly returns the same hint as the full
coupling (the hint path is shared - only the cadence formula differs).

Anchors for these baselines (post-2020 only):
  SAPACS: Wang X. et al. 2025, Springer Wireless Networks.
  MP-TMD: cluster of trust-based misbehaviour-detection schemes pre-2024;
          summarised in Babaghayou et al. 2025 ACM CSUR.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..lifecycle.coupling import (
    CouplingConfig,
    CouplingLayer,
    NeighbourhoodSummary,
    PeerState,
    _clamp,
)


# ------------------ SAPACS-style: context only ------------------

@dataclass(frozen=True)
class ContextOnlyConfig:
    """SAPACS-style: cadence adapts on density alone."""
    C_base: float = 1.0 / 60.0
    C_min: float = 1.0 / 300.0
    C_max: float = 1.0 / 10.0
    gamma_rho: float = 1.5            # SAPACS uses stronger density gain since
                                       # it has no other axis to discriminate
    rho_low: float = 5.0
    rho_high: float = 30.0


class ContextOnlyCoupling:
    """Rotation cadence is a function of local density only.

    Cannot distinguish benign-dense (motorway during rush hour) from
    hostile-dense (Sybil cluster in an urban grid) - this is the gap our
    closed-loop coupling closes.
    """
    def __init__(self, config: ContextOnlyConfig | None = None) -> None:
        self.cfg = config or ContextOnlyConfig()

    def cadence(self, neighbourhood: NeighbourhoodSummary) -> float:
        cfg = self.cfg
        if cfg.rho_high <= cfg.rho_low:
            return cfg.C_base
        normalised = _clamp(0.0, 1.0,
                            (neighbourhood.rho_t - cfg.rho_low) /
                            (cfg.rho_high - cfg.rho_low))
        f = 1.0 + cfg.gamma_rho * normalised
        return _clamp(cfg.C_min, cfg.C_max, cfg.C_base * f)

    def revocation_hint(
        self, peer: PeerState, neighbourhood: NeighbourhoodSummary | None = None
    ) -> float:
        # SAPACS does not consume detector output; no hint channel.
        return 0.0


# ------------------ MP-TMD-style: trust only ------------------

class TrustOnlyCoupling:
    """The earlier trust-only form of our coupling: trust-driven, no physics.

    Identical to `CouplingLayer` except g_density and g_churn are forced
    to 1.0. Hint surface is identical to the full framework (this
    captures most trust-based misbehaviour-detection prior art).
    """
    def __init__(self, config: CouplingConfig | None = None) -> None:
        self.cfg = config or CouplingConfig()

    def cadence(self, neighbourhood: NeighbourhoodSummary) -> float:
        cfg = self.cfg
        if neighbourhood.n < cfg.n_min:
            return cfg.C_base
        trust_term = max(0.0, cfg.theta_safe - neighbourhood.mean)
        f = 1.0 + cfg.alpha * trust_term
        return _clamp(cfg.C_min, cfg.C_max, cfg.C_base * f)

    def revocation_hint(
        self, peer: PeerState, neighbourhood: NeighbourhoodSummary | None = None
    ) -> float:
        cfg = self.cfg
        if peer.msg_count < cfg.cold_start_msgs:
            return 0.0
        if peer.T_peer >= cfg.theta_revoke:
            return 0.0
        if peer.count_below < cfg.k:
            return 0.0
        diff = cfg.theta_revoke - peer.T_peer
        return _clamp(0.0, 1.0, diff ** cfg.beta)


# ------------------ No coupling: floor ------------------

@dataclass(frozen=True)
class NoCouplingConfig:
    C_base: float = 1.0 / 60.0


class NoCoupling:
    """No adaptive behaviour. Cadence fixed at C_base; no hints. The floor."""
    def __init__(self, config: NoCouplingConfig | None = None) -> None:
        self.cfg = config or NoCouplingConfig()

    def cadence(self, neighbourhood: NeighbourhoodSummary) -> float:
        return self.cfg.C_base

    def revocation_hint(
        self, peer: PeerState, neighbourhood: NeighbourhoodSummary | None = None
    ) -> float:
        return 0.0


# ------------------ Churn-hint extension ------------------

@dataclass(frozen=True)
class ChurnHintConfig:
    """Configuration for the trust-independent churn hint path.

    The churn-hint fires for cold-start peers whenever the neighbourhood
    pseudonym-churn rate `kappa_t` exceeds `kappa_threshold`. Unlike the
    trust-based revocation_hint, it does NOT require the per-peer trust
    signal to be below `theta_revoke` - so it survives attackers (e.g.
    A-PHY-1) who defeat the per-message detector.

    Defaults targeted at the rotation_dos scenario, which spawns 20 sybils
    simultaneously: `kappa_threshold = 3.0` flags any neighbourhood seeing
    more than 3 fresh pseudonyms per (aggregator window) duration.
    """
    kappa_threshold: float = 3.0
    cold_start_msgs: int = 25
    hint_strength: float = 0.6


class FullDD007ChurnHint(CouplingLayer):
    """Full coupling + trust-independent churn hint path.

    Inherits cadence() and trust-based revocation_hint() from CouplingLayer
    unchanged. Adds churn_hint(peer, neighbourhood) which fires when:

        peer.msg_count <  cold_start_msgs         (peer is fresh)
        AND
        neighbourhood.kappa_t > kappa_threshold   (neighbourhood is churning)

    The simulator merges churn_hint and revocation_hint via max(), so the
    overall hint is the union of trust-based and churn-based signals.
    """

    def __init__(
        self,
        config: CouplingConfig | None = None,
        churn_config: ChurnHintConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.churn_cfg = churn_config or ChurnHintConfig()

    def churn_hint(
        self, peer: PeerState, neighbourhood: NeighbourhoodSummary
    ) -> float:
        cfg = self.churn_cfg
        if peer.msg_count >= cfg.cold_start_msgs:
            return 0.0
        if neighbourhood.kappa_t <= cfg.kappa_threshold:
            return 0.0
        excess = neighbourhood.kappa_t - cfg.kappa_threshold
        normalised = _clamp(0.0, 1.0, excess / max(cfg.kappa_threshold, 1e-9))
        return _clamp(0.0, 1.0, cfg.hint_strength * normalised)


# ------------------ Cumulative-density hint extension ------------------

@dataclass(frozen=True)
class DensityHintConfig:
    """Configuration for the cumulative-density hint path.

    CL-3 identified that A-PHY-1 Sybil attackers staggering arrivals at
    `inter_arrival_s >= 2s` defeat the churn-hint (kappa_t stays at
    baseline). The density-hint targets that gap by tracking the TOTAL
    unique pseudonyms observed by this node, regardless of instantaneous
    arrival rate.

    Fires when:
      n_total_unique > unique_threshold              (population anomalous)
      AND
      (now - first_seen[pk]) <= recent_window_s      (peer is "recent")

    The recent-window filter prevents the hint from firing on settled
    honest peers; the cumulative threshold catches the slow-arrival
    Sybil cohort that the kappa_t-based churn-hint misses.
    """
    unique_threshold: int = 8
    recent_window_s: float = 12.0
    hint_strength: float = 0.6


class FullDD007DensityHint(FullDD007ChurnHint):
    """Full coupling + churn-hint + cumulative-density hint.

    Inherits cadence(), trust-based revocation_hint(), and the
    churn_hint() from FullDD007ChurnHint unchanged. Adds an internal
    per-pseudonym first_seen registry consumed by density_hint().

    The simulator calls density_hint via duck-typing and merges its
    output into the final r_p via max().
    """

    def __init__(
        self,
        config: CouplingConfig | None = None,
        churn_config: ChurnHintConfig | None = None,
        density_config: DensityHintConfig | None = None,
    ) -> None:
        super().__init__(config, churn_config)
        self.density_cfg = density_config or DensityHintConfig()
        # Internal first-seen registry. Per-instance, not shared.
        self._first_seen: dict[object, float] = {}

    def density_hint(
        self,
        peer: PeerState,
        neighbourhood: NeighbourhoodSummary,
        *,
        pk,
        now: float,
    ) -> float:
        cfg = self.density_cfg
        if pk not in self._first_seen:
            self._first_seen[pk] = now

        n_total_unique = len(self._first_seen)
        if n_total_unique <= cfg.unique_threshold:
            return 0.0

        # Only fire for peers that are themselves "recent". Settled honest
        # peers fall outside the recent window and stay silent.
        if (now - self._first_seen[pk]) > cfg.recent_window_s:
            return 0.0

        excess = n_total_unique - cfg.unique_threshold
        normalised = _clamp(0.0, 1.0, excess / max(cfg.unique_threshold, 1))
        return _clamp(0.0, 1.0, cfg.hint_strength * normalised)


# ------------------ Naive threshold-based revocation (Phase 1 baseline) ------------------

@dataclass(frozen=True)
class NaiveThresholdConfig:
    """Naive detection-only baseline: revoke when aggregated trust drops
    below threshold. No cold-start gating, no persistence requirement,
    no cross-layer signals.

    Represents a literature baseline pattern where detector alerts drive
    revocation directly without any coupling-layer policy: e.g. Kamel 2020,
    Babaghayou 2025, and the broader 'classifier outputs binary verdict,
    receiver revokes on verdict' family.
    """
    revoke_threshold: float = 0.5
    c_base: float = 1.0 / 60.0


class NaiveThresholdRevocation:
    """Detection-only baseline with instant trust-threshold revocation.

    cadence(): fixed at c_base (no cadence adaptation).
    revocation_hint(): 1.0 when peer.T_peer < revoke_threshold else 0.0.

    Compared to TrustOnlyCoupling, this variant removes:
      - the cold-start gate (cold_start_msgs)
      - the persistence requirement (k consecutive low-trust observations)
      - the (theta_revoke - T_peer)^beta hint magnitude shaping

    Maps to a published baseline where a per-message classifier output
    drives revocation directly, with no coupling-layer policy mediating.
    """

    def __init__(self, config: NaiveThresholdConfig | None = None) -> None:
        self.cfg = config or NaiveThresholdConfig()

    def cadence(self, neighbourhood: NeighbourhoodSummary) -> float:
        return self.cfg.c_base

    def revocation_hint(
        self, peer: PeerState, neighbourhood: NeighbourhoodSummary | None = None
    ) -> float:
        if peer.T_peer < self.cfg.revoke_threshold:
            return 1.0
        return 0.0


# ------------------ Registry ------------------

COUPLING_VARIANTS = {
    "no_coupling":          NoCoupling,
    "context_only":         ContextOnlyCoupling,
    "trust_only":           TrustOnlyCoupling,
    "full_dd007":           CouplingLayer,
    "full_dd007_churn":     FullDD007ChurnHint,
    "full_dd007_density":   FullDD007DensityHint,
    "naive_revocation":     NaiveThresholdRevocation,
}
