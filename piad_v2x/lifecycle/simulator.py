"""End-to-end streaming simulator.

Wires the pipeline stages into a single message-by-message flow.

    msg --> StreamingFeatureExtractor
        --> TrustProvider (or oracle)
        --> TrustAggregator
        --> CouplingLayer
        --> MockPseudonymManager

The simulator is deliberately independent of any specific TrustProvider so
that the streaming, aggregation, coupling, and pseudonym-manager logic can
be verified against analytical bounds (e.g. the D1 / P1 2.5 s revocation-
hint window) without depending on the PINN's training quality.

Two TrustProvider implementations ship here:

- OracleTrustProvider: trust is a function of the ground-truth class label.
  Used for D1 timing-bound verification (we KNOW class 0 = benign, etc).
- PINNTrustProvider:   trust comes from a trained PINNDetector. Used
  when the simulator runs as the full deployed stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Protocol, runtime_checkable

import numpy as np

from .aggregator import TrustAggregator
from .coupling import CouplingLayer
from ..data_utils.feature_extractor import FEATURE_NAMES, FeatureConfig
from .manager import MockPseudonymManager


# ---------------- streaming feature extractor ----------------

class StreamingFeatureExtractor:
    """One-message-at-a-time variant of `feature_extractor.extract_features`.

    Keeps per-pseudonym last-message state so kinematic deltas and residuals
    can be computed on the fly. Same cold-start semantics as the batch path:
    first message per pseudonym is cold; gaps > max_dt_seconds reset to cold.
    """
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.cfg = config or FeatureConfig()
        self._last: dict[Hashable, dict] = {}

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES

    def process(self, row: dict) -> np.ndarray:
        """Process one message dict and return a (len(FEATURE_NAMES),) array.

        Required keys: senderPseudo, sendTime, posx, posy, spdx, spdy,
                       aclx, acly, hedx, hedy.
        """
        cfg = self.cfg
        pk = row["senderPseudo"]
        t = float(row["sendTime"])
        posx = float(row["posx"]); posy = float(row["posy"])
        spdx = float(row["spdx"]); spdy = float(row["spdy"])
        aclx = float(row["aclx"]); acly = float(row["acly"])
        hedx = float(row["hedx"]); hedy = float(row["hedy"])

        prev = self._last.get(pk)
        if prev is None:
            d_t = 0.0
            cold = True
            d_posx = d_posy = 0.0
            d_spdx = d_spdy = d_hedx = d_hedy = 0.0
        else:
            d_t = t - prev["t"]
            if d_t <= cfg.min_dt_seconds or d_t > cfg.max_dt_seconds:
                cold = True
                d_posx = d_posy = 0.0
                d_spdx = d_spdy = d_hedx = d_hedy = 0.0
            else:
                cold = False
                d_posx = posx - prev["posx"]
                d_posy = posy - prev["posy"]
                d_spdx = spdx - prev["spdx"]
                d_spdy = spdy - prev["spdy"]
                d_hedx = hedx - prev["hedx"]
                d_hedy = hedy - prev["hedy"]

        safe_dt = max(d_t, cfg.min_dt_seconds)
        d_spd_mag = float(np.hypot(d_spdx, d_spdy))
        d_hed_mag = float(np.hypot(d_hedx, d_hedy))
        v_star = float(np.hypot(d_posx, d_posy)) / safe_dt if not cold else 0.0
        spd_reported = float(np.hypot(spdx, spdy))
        acl_reported = float(np.hypot(aclx, acly))
        a_star = d_spd_mag / safe_dt if not cold else 0.0
        r_star = d_hed_mag / safe_dt if not cold else 0.0
        e_v = (v_star - spd_reported) if not cold else 0.0
        e_a = (a_star - acl_reported) if not cold else 0.0

        if cold:
            d_posx = d_posy = 0.0
            d_spd_mag = d_hed_mag = d_t = 0.0

        self._last[pk] = {
            "t": t, "posx": posx, "posy": posy,
            "spdx": spdx, "spdy": spdy,
            "hedx": hedx, "hedy": hedy,
        }

        # S1 physics-plausibility flags (mirror feature_extractor batch logic).
        def _m(x: float, b: float) -> float:
            return min(max(abs(x) / b - 1.0, 0.0), 1.0)

        p_spd_lim = _m(spd_reported, cfg.v_max_mps)
        p_acl_lim = _m(acl_reported, cfg.a_max_mps2)
        p_posjump = _m(v_star, cfg.v_max_mps)         # v_star is 0 on cold rows
        p_spd_cons = _m(e_v, cfg.tau_v_consistency)   # e_v is 0 on cold rows
        motion_norm = float(np.hypot(d_posx, d_posy))
        hed_norm = float(np.hypot(hedx, hedy))
        if (not cold) and motion_norm > cfg.min_dt_seconds \
                and hed_norm > cfg.min_dt_seconds and v_star > cfg.min_speed_for_heading:
            cos_sim = (d_posx * hedx + d_posy * hedy) / (motion_norm * hed_norm)
            p_hed_cons = min(max((1.0 - cos_sim) / 2.0, 0.0), 1.0)
        else:
            p_hed_cons = 0.0

        return np.array([
            posx, posy, spdx, spdy, aclx, acly, hedx, hedy,
            d_posx, d_posy, d_spd_mag, d_hed_mag, d_t,
            v_star, a_star, r_star,
            e_v, e_a,
            1.0 if cold else 0.0,
            p_spd_lim, p_acl_lim, p_posjump, p_spd_cons, p_hed_cons,
        ], dtype=np.float64)


# ---------------- trust providers ----------------

@runtime_checkable
class TrustProvider(Protocol):
    def trust(self, x: np.ndarray, row: dict) -> float: ...


@dataclass
class OracleTrustProvider:
    """Ground-truth-class-conditioned trust. Used to isolate the lifecycle from detector quality."""
    benign_trust: float = 0.95
    attack_trust: float = 0.05
    class_to_trust: dict[int, float] = field(default_factory=dict)

    def trust(self, x: np.ndarray, row: dict) -> float:
        cls = int(row.get("class", 0))
        if cls in self.class_to_trust:
            return self.class_to_trust[cls]
        return self.benign_trust if cls == 0 else self.attack_trust


class PINNTrustProvider:
    """Wraps a trained PINNDetector for single-sample trust evaluation."""
    def __init__(self, model, device=None) -> None:
        import torch
        self.torch = torch
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.eval()
        self.model.to(self.device)

    def trust(self, x: np.ndarray, row: dict) -> float:
        torch = self.torch
        with torch.no_grad():
            xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32) \
                      .unsqueeze(0).to(self.device)
            _, t = self.model(xt)
        return float(t.item())


# ---------------- end-to-end stream ----------------

@dataclass
class StreamReport:
    n_messages: int
    n_pseudonyms: int
    n_hint_events: int
    n_rotation_events: int
    duration_seconds: float
    first_hint_time: dict[Hashable, float] = field(default_factory=dict)
    # D7 physics-violation flag history.
    # Each entry is (time, drho_dt, phi_t_fired).
    physics_flag_events: list[tuple[float, float, bool]] = field(default_factory=list)
    n_physics_flags: int = 0
    # Cadence trace for analysing C_max engagement under DoS scenarios.
    cadence_trace: list[tuple[float, float]] = field(default_factory=list)
    # Number of per-message classification frame events (F1 mitigation).
    n_frame_events: int = 0


def stream_messages(
    rows: Iterable[dict],
    trust_provider: TrustProvider,
    extractor: StreamingFeatureExtractor | None = None,
    aggregator: TrustAggregator | None = None,
    coupling: CouplingLayer | None = None,
    manager: MockPseudonymManager | None = None,
    phi_t_threshold: float = 5.0,
    record_cadence_trace: bool = False,
    frame_event_threshold: float | None = None,
) -> tuple[StreamReport, MockPseudonymManager]:
    """Run the full streaming pipeline over an iterable of message dicts.

    Returns (report, manager). The manager is the live recording surface.
    """
    extractor = extractor or StreamingFeatureExtractor()
    aggregator = aggregator or TrustAggregator()
    coupling = coupling or CouplingLayer()
    manager = manager or MockPseudonymManager()

    n_messages = 0
    seen_pseudonyms: set[Hashable] = set()
    t_first = float("inf")
    t_last = float("-inf")
    physics_flag_events: list[tuple[float, float, bool]] = []
    n_physics_flags = 0
    cadence_trace: list[tuple[float, float]] = []

    for row in rows:
        x = extractor.process(row)
        s_t = trust_provider.trust(x, row)
        t = float(row["sendTime"])
        pk = row["senderPseudo"]

        # per-peer aggregation
        peer_state = aggregator.update(pk, s_t, now=t)

        # F1 stealth-attacker mitigation: per-message classification stream.
        # When the per-message trust score is below frame_event_threshold,
        # record a misbehaviour frame event independent of the cumulative
        # revocation hint path. (Hint path requires persistence + cold-start
        # gating; this stream does not, by design.)
        if frame_event_threshold is not None and s_t < frame_event_threshold:
            manager.submit_frame_event(pk, p_misbehaviour=1.0 - s_t, now=t)

        # compute neighbourhood once, use it for both cadence and the
        # optional churn-hint path (duck-typed; absent on legacy variants).
        neighbourhood = aggregator.neighbourhood(now=t)

        # pass the neighbourhood so CouplingLayer can physics-
        # condition theta_revoke. Inert unless gamma_revoke > 0 (default 0.0);
        # baselines accept and ignore it. So this is a no-op when the gate is off.
        r_trust = coupling.revocation_hint(peer_state, neighbourhood)
        r_p = r_trust
        # trust-independent churn hint (kappa-driven, instantaneous)
        churn_hint = getattr(coupling, "churn_hint", None)
        if churn_hint is not None:
            r_p = max(r_p, churn_hint(peer_state, neighbourhood))
        # trust-independent density hint (cumulative-unique-driven)
        density_hint = getattr(coupling, "density_hint", None)
        if density_hint is not None:
            r_p = max(r_p, density_hint(peer_state, neighbourhood, pk=pk, now=t))
        manager.submit_hint(pk, r_p, now=t)

        c_self = coupling.cadence(neighbourhood)
        manager.set_rotation_rate(c_self, now=t)

        # D7 / phi_t physics-violation flag: phantom-mass detection from
        # the density gradient. Fires when ∂ρ/∂t exceeds a plausible-arrival
        # threshold (vehicles appearing without entering the segment).
        phi = bool(neighbourhood.drho_dt > phi_t_threshold)
        physics_flag_events.append((t, neighbourhood.drho_dt, phi))
        if phi:
            n_physics_flags += 1

        if record_cadence_trace:
            cadence_trace.append((t, c_self))

        n_messages += 1
        seen_pseudonyms.add(pk)
        if t < t_first:
            t_first = t
        if t > t_last:
            t_last = t

    report = StreamReport(
        n_messages=n_messages,
        n_pseudonyms=len(seen_pseudonyms),
        n_hint_events=manager.n_hints(),
        n_rotation_events=manager.n_rotation_events(),
        duration_seconds=max(0.0, t_last - t_first) if n_messages else 0.0,
        first_hint_time={pk: manager.first_hint_time(pk) for pk in seen_pseudonyms
                         if manager.first_hint_time(pk) is not None},
        physics_flag_events=physics_flag_events,
        n_physics_flags=n_physics_flags,
        cadence_trace=cadence_trace,
        n_frame_events=manager.n_frame_events(),
    )
    return report, manager
