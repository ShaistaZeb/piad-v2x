"""Mock pseudonym manager.

Implements the pseudonym-manager surface only - the framework
does not own pseudonym lifecycle; this thin adapter records the directives
that an unmodified IEEE 1609.2 pseudonym manager would consume.

The full surface is two calls:

    set_rotation_rate(rate)    - clamped externally to [C_min, C_max]
    submit_hint(peer, R_p)     - appends to the local pre-revocation register

PIAD-V2X never executes a revocation. The MA is authoritative.
This module is a recording surface for end-to-end simulation + property tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable


@dataclass
class HintEvent:
    time: float
    peer: Hashable
    hint: float


@dataclass
class RotationEvent:
    time: float
    rate: float


@dataclass
class MisbehaviourFrameEvent:
    """Per-message classification evidence (the false-negative mitigation).

    Distinct from `HintEvent` which is the *cumulative* revocation hint
    R_p(t). Frame events are *per-message* classifications c_t with a
    misbehaviour probability above a configured threshold. A stealth
    attacker who keeps EMA-trust above θ_revoke can still produce frame
    events because each individual implausible message is classified
    independently.
    """
    time: float
    peer: Hashable
    p_misbehaviour: float


@dataclass
class MockPseudonymManager:
    """Recording adapter. No state machine; the inputs are facts as observed."""
    current_rotation_rate: float = 0.0
    rotation_history: list[RotationEvent] = field(default_factory=list)
    hint_history: list[HintEvent] = field(default_factory=list)
    frame_history: list[MisbehaviourFrameEvent] = field(default_factory=list)
    _first_hint_time: dict[Hashable, float] = field(default_factory=dict)
    _first_frame_time: dict[Hashable, float] = field(default_factory=dict)

    def set_rotation_rate(self, rate: float, now: float = 0.0) -> None:
        """Record a cadence directive. Caller (the coupling layer) is
        responsible for clamping to [C_min, C_max]; this adapter records what
        it is told."""
        self.current_rotation_rate = rate
        self.rotation_history.append(RotationEvent(time=now, rate=rate))

    def submit_hint(self, peer: Hashable, hint: float, now: float = 0.0) -> None:
        """Record a non-zero revocation-hint contribution for a peer.

        Zero or negative hints are dropped (R_p = 0 means no evidence).
        First-hint timestamp per peer is cached for D1 timing checks.
        """
        if hint <= 0.0:
            return
        self.hint_history.append(HintEvent(time=now, peer=peer, hint=hint))
        if peer not in self._first_hint_time:
            self._first_hint_time[peer] = now

    # --- introspection helpers (used by simulator + tests) ---

    def first_hint_time(self, peer: Hashable) -> float | None:
        return self._first_hint_time.get(peer)

    def hints_for(self, peer: Hashable) -> list[HintEvent]:
        return [h for h in self.hint_history if h.peer == peer]

    def n_hints(self) -> int:
        return len(self.hint_history)

    def n_rotation_events(self) -> int:
        return len(self.rotation_history)

    def submit_frame_event(
        self, peer: Hashable, p_misbehaviour: float, now: float = 0.0,
    ) -> None:
        """Record a per-message classification verdict above-threshold.

        This is the F1 stealth-attacker mitigation surface: even if the
        cumulative R_p hint does not fire, individual frames may.
        """
        if p_misbehaviour <= 0.0:
            return
        self.frame_history.append(
            MisbehaviourFrameEvent(time=now, peer=peer, p_misbehaviour=p_misbehaviour)
        )
        if peer not in self._first_frame_time:
            self._first_frame_time[peer] = now

    def first_frame_time(self, peer: Hashable) -> float | None:
        return self._first_frame_time.get(peer)

    def frames_for(self, peer: Hashable) -> list[MisbehaviourFrameEvent]:
        return [e for e in self.frame_history if e.peer == peer]

    def n_frame_events(self) -> int:
        return len(self.frame_history)
