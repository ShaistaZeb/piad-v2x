"""Trust aggregator.

Maintains short-window EMA trust per peer pseudonym, plus a physics-informed
neighbourhood summary for the coupling layer.

The aggregator surfaces three physics-grounded quantities through the neighbourhood
summary in addition to the trust statistics:

- ρ_t       : local vehicle density (peers per coverage area).
- ∂ρ/∂t     : density gradient over a 2-second window (finite difference).
- κ_t       : pseudonym churn rate (new pseudonyms first-seen, per second,
              over the neighbourhood window w_t).

These feed g_density and g_churn in CouplingLayer.cadence().
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

from .coupling import NeighbourhoodSummary, PeerState


@dataclass(frozen=True)
class AggregatorConfig:
    window: int = 50
    alpha: float = 0.85
    theta_low: float = 0.5
    eviction_seconds: float = 10.0
    neighbourhood_seconds: float = 5.0   # w_t
    cold_start_T: float = 0.5

    # Physics-informed additions.
    # ρ_t = (active peer count) / coverage_area. In this prototype coverage_area
    # is a unit normalisation; downstream g_density thresholds (rho_low=5,
    # rho_high=30) are expressed in the same units. Real-world deployments can
    # set coverage_area to the radio footprint area in m^2.
    coverage_area: float = 1.0
    gradient_window_seconds: float = 2.0  # finite-difference span for ∂ρ/∂t


@dataclass
class _PeerInternal:
    first_seen: float
    last_seen: float
    T_peer: float
    msg_count: int = 0
    window: Deque[float] = field(default_factory=deque)
    count_below: int = 0


class TrustAggregator:
    def __init__(self, config: AggregatorConfig | None = None) -> None:
        self.cfg = config or AggregatorConfig()
        self._peers: Dict[str, _PeerInternal] = {}
        # Physics-informed internal state.
        # _density_history: chronologically ordered (timestamp, n_active_at_t)
        # snapshots, used for the finite-difference ∂ρ/∂t estimate.
        self._density_history: Deque[Tuple[float, int]] = deque()
        # _first_seen_log: timestamps at which a previously unseen pseudonym
        # was first observed. Used to compute κ_t over the last w_t seconds.
        self._first_seen_log: Deque[float] = deque()

    def update(
        self,
        pseudonym: str,
        trust_score: float,
        now: float | None = None,
    ) -> PeerState:
        if now is None:
            now = time.monotonic()
        peer = self._peers.get(pseudonym)
        is_new = peer is None
        if is_new:
            peer = _PeerInternal(
                first_seen=now,
                last_seen=now,
                T_peer=self.cfg.cold_start_T,
            )
            self._peers[pseudonym] = peer
            self._first_seen_log.append(now)

        peer.last_seen = now
        peer.msg_count += 1
        peer.T_peer = (
            self.cfg.alpha * peer.T_peer
            + (1.0 - self.cfg.alpha) * trust_score
        )
        peer.window.append(trust_score)
        if len(peer.window) > self.cfg.window:
            peer.window.popleft()
        peer.count_below = sum(1 for s in peer.window if s < self.cfg.theta_low)

        # snapshot the current active count for ∂ρ/∂t lookups.
        n_active = self._active_count(now)
        self._density_history.append((now, n_active))
        self._evict_density_history(now)
        self._evict_first_seen_log(now)

        return self._snapshot(peer)

    def evict(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.cfg.eviction_seconds
        self._peers = {p: s for p, s in self._peers.items() if s.last_seen >= cutoff}

    def neighbourhood(self, now: float | None = None) -> NeighbourhoodSummary:
        if now is None:
            now = time.monotonic()
        self._evict_density_history(now)
        self._evict_first_seen_log(now)

        active = sorted(
            s.T_peer for s in self._peers.values()
            if s.last_seen >= now - self.cfg.neighbourhood_seconds
        )
        n_active = len(active)
        if not active:
            return NeighbourhoodSummary(
                mean=self.cfg.cold_start_T,
                q25=self.cfg.cold_start_T,
                n=0,
                rho_t=0.0,
                drho_dt=0.0,
                kappa_t=self._churn_rate(now),
            )

        mean = statistics.fmean(active)
        q25_idx = max(0, (len(active) - 1) // 4)
        q25 = active[q25_idx]

        rho_t = float(n_active) / self.cfg.coverage_area
        drho_dt = self._density_gradient(now, rho_t)
        kappa_t = self._churn_rate(now)

        return NeighbourhoodSummary(
            mean=mean,
            q25=q25,
            n=n_active,
            rho_t=rho_t,
            drho_dt=drho_dt,
            kappa_t=kappa_t,
        )

    def peer_states(self) -> Dict[str, PeerState]:
        return {p: self._snapshot(s) for p, s in self._peers.items()}

    # --- physics-modulator internals ---

    def _active_count(self, now: float) -> int:
        cutoff = now - self.cfg.neighbourhood_seconds
        return sum(1 for s in self._peers.values() if s.last_seen >= cutoff)

    def _density_gradient(self, now: float, rho_now: float) -> float:
        """∂ρ/∂t estimated by finite difference over `gradient_window_seconds`.

        Picks the most recent density snapshot at or before (now - gradient_window).
        Returns 0.0 if no snapshot that old exists yet (early-life cold start).
        """
        target_t = now - self.cfg.gradient_window_seconds
        candidate: Tuple[float, int] | None = None
        for ts, n in self._density_history:
            if ts <= target_t:
                candidate = (ts, n)
            else:
                break
        if candidate is None:
            return 0.0
        ts_old, n_old = candidate
        rho_old = float(n_old) / self.cfg.coverage_area
        dt = now - ts_old
        if dt <= 0.0:
            return 0.0
        return (rho_now - rho_old) / dt

    def _churn_rate(self, now: float) -> float:
        """κ_t: distinct new pseudonyms first-seen in the last w_t, per second."""
        w_t = self.cfg.neighbourhood_seconds
        if w_t <= 0:
            return 0.0
        cutoff = now - w_t
        new_count = sum(1 for ts in self._first_seen_log if ts >= cutoff)
        return float(new_count) / w_t

    def _evict_density_history(self, now: float) -> None:
        # Keep enough history that a gradient lookup at target_t still finds
        # an entry. Retain slightly more than the gradient window.
        retain = max(self.cfg.gradient_window_seconds * 2.0, self.cfg.neighbourhood_seconds)
        cutoff = now - retain
        while self._density_history and self._density_history[0][0] < cutoff:
            self._density_history.popleft()

    def _evict_first_seen_log(self, now: float) -> None:
        cutoff = now - self.cfg.neighbourhood_seconds
        while self._first_seen_log and self._first_seen_log[0] < cutoff:
            self._first_seen_log.popleft()

    @staticmethod
    def _snapshot(peer: _PeerInternal) -> PeerState:
        return PeerState(
            msg_count=peer.msg_count,
            T_peer=peer.T_peer,
            count_below=peer.count_below,
        )
