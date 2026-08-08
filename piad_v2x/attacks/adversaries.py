"""Synthetic adversary generators for case studies.

Each generator yields message dicts in the shape the streaming simulator
expects (see simulator.StreamingFeatureExtractor.process), with `class`
set so the OracleTrustProvider produces an appropriate trust score.

The four archetypes:

- persistent_attacker:        peer consistently low trust
- stealthy_attacker:          interleaves benign frames
- coordinated_rotation_dos:   Sybil cluster forces cadence up
- cold_start_gaming:          rotates inside w/2 cold-start

A small population of honest peers can be added to any scenario via
`honest_traffic`, so neighbourhood statistics (ρ_t, κ_t, N_t.mean) are
realistic and the g_density / g_churn modulators see meaningful input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------- low-level helper ----------------

def _msg(
    *, pk: int, sender: int, t: float, cls: int,
    posx: float = 0.0, posy: float = 0.0,
    spdx: float = 1.0, spdy: float = 0.0,
    aclx: float = 0.0, acly: float = 0.0,
    hedx: float = 1.0, hedy: float = 0.0,
) -> dict:
    return {
        "senderPseudo": pk, "sender": sender, "sendTime": t, "class": cls,
        "posx": posx, "posy": posy,
        "spdx": spdx, "spdy": spdy,
        "aclx": aclx, "acly": acly,
        "hedx": hedx, "hedy": hedy,
    }


# ---------------- traffic generators ----------------

def honest_traffic(
    n_peers: int = 10,
    duration_s: float = 30.0,
    rate_hz: float = 10.0,
    start_pk: int = 1000,
    start_t: float = 0.0,
    speed: float = 12.0,
) -> list[dict]:
    """Honest peers cruising at a constant speed. All class=0 (benign)."""
    out: list[dict] = []
    n_frames = int(duration_s * rate_hz)
    for pk_offset in range(n_peers):
        pk = start_pk + pk_offset
        x0 = float(pk_offset * 30.0)
        for i in range(n_frames):
            t = start_t + i / rate_hz
            out.append(_msg(
                pk=pk, sender=pk, t=t, cls=0,
                posx=x0 + speed * (t - start_t), posy=0.0,
                spdx=speed, spdy=0.0,
            ))
    return out


def persistent_attacker(
    pk: int = 9001,
    duration_s: float = 30.0,
    rate_hz: float = 10.0,
    start_t: float = 0.0,
    attack_class: int = 5,
) -> list[dict]:
    """Single peer, every message implausible. Spec §5.1.

    Verifies D1 / P1: first revocation hint within max(w/2, k) messages.
    """
    out: list[dict] = []
    n_frames = int(duration_s * rate_hz)
    for i in range(n_frames):
        t = start_t + i / rate_hz
        out.append(_msg(
            pk=pk, sender=pk, t=t, cls=attack_class,
            posx=100.0 + i * 0.5, posy=0.0,
            spdx=1.0, spdy=0.0,
        ))
    return out


def stealthy_attacker(
    pk: int = 9002,
    duration_s: float = 30.0,
    rate_hz: float = 10.0,
    start_t: float = 0.0,
    attack_class: int = 5,
    burst_attack: int = 3,
    burst_benign: int = 7,
) -> list[dict]:
    """Single peer interleaving implausible bursts with benign frames. Spec §5.2.

    Default 3-attack / 7-benign cycle keeps the EMA-trust above θ_revoke=0.3
    while still firing attack-shaped messages periodically. Tests whether
    the strict revocation gate (theta_revoke + persistence k) blocks the
    hint, and whether the φ_t flag could catch what the trust path misses.
    """
    out: list[dict] = []
    n_frames = int(duration_s * rate_hz)
    cycle = burst_attack + burst_benign
    for i in range(n_frames):
        t = start_t + i / rate_hz
        is_attack = (i % cycle) < burst_attack
        out.append(_msg(
            pk=pk, sender=pk, t=t,
            cls=(attack_class if is_attack else 0),
            posx=200.0 + i * 0.5, posy=0.0,
            spdx=1.0, spdy=0.0,
        ))
    return out


def coordinated_rotation_dos(
    n_sybils: int = 20,
    duration_s: float = 10.0,
    rate_hz: float = 10.0,
    start_pk: int = 50000,
    start_t: float = 0.0,
    attack_class: int = 5,
) -> list[dict]:
    """N pseudonyms, all attack, sustained simultaneously. Spec §5.3.

    Aims at the cadence layer: pushes N_t.mean down, drives C_self toward
    C_max. With the physics-conditioned coupling active, g_churn should damp the response because
    κ_t (new pseudonyms in last w_t) is high. C_max is the hard ceiling
    that prevents unbounded rotation-DoS.
    """
    out: list[dict] = []
    n_frames = int(duration_s * rate_hz)
    for s_off in range(n_sybils):
        pk = start_pk + s_off
        x0 = float(s_off * 5.0)
        for i in range(n_frames):
            t = start_t + i / rate_hz
            out.append(_msg(
                pk=pk, sender=pk, t=t, cls=attack_class,
                posx=x0 + i * 0.5, posy=0.0,
                spdx=1.0, spdy=0.0,
            ))
    return out


def staggered_rotation_dos(
    n_sybils: int = 20,
    inter_arrival_s: float = 1.0,
    duration_s: float = 30.0,
    rate_hz: float = 10.0,
    start_pk: int = 60000,
    attack_class: int = 5,
) -> list[dict]:
    """N pseudonyms arriving STAGGERED to defeat churn-hint detection.

    Unlike `coordinated_rotation_dos` (all sybils start at t=0), this
    constructor staggers sybil first-appearances at `inter_arrival_s`
    seconds apart. With default `inter_arrival_s=1.0`, kappa_t never
    spikes - the rate of new pseudonyms remains close to the benign
    baseline. Each sybil persists from its appearance until duration_s.

    Designed to test the boundary of the churn-hint path: does the
    framework retain hint-firing capability against a slow-stagger
    attacker?
    """
    out: list[dict] = []
    for s_off in range(n_sybils):
        pk = start_pk + s_off
        t_start = s_off * inter_arrival_s
        if t_start >= duration_s:
            break
        n_frames = int((duration_s - t_start) * rate_hz)
        x0 = float(s_off * 5.0)
        for i in range(n_frames):
            t = t_start + i / rate_hz
            out.append(_msg(
                pk=pk, sender=pk, t=t, cls=attack_class,
                posx=x0 + i * 0.5, posy=0.0,
                spdx=1.0, spdy=0.0,
            ))
    return out


def staged_retirement_sybil(
    n_concurrent: int = 3,
    rotation_period_s: float = 5.0,
    duration_s: float = 25.0,
    rate_hz: float = 10.0,
    start_pk: int = 80000,
    attack_class: int = 5,
) -> list[dict]:
    """Bounded concurrent Sybils with periodic pseudonym retirement.

    Maintains exactly `n_concurrent` active pseudonyms at any time. Every
    `rotation_period_s` seconds, each active pseudonym is retired (stops
    broadcasting) and a fresh pseudonym takes its place. Total unique
    pseudonyms over the simulation = n_concurrent * ceil(duration_s /
    rotation_period_s).

    Designed to test whether the cumulative-density-hint can be
    evaded by an attacker who limits *concurrent* pseudonym count to
    stay below threshold, while rotating identities to refresh the
    attack window. Because the defender's `_first_seen` registry is
    monotonic, total unique grows linearly in number of rotations - so
    this attack only evades when (rotations * n_concurrent + honest)
    stays below unique_threshold.

    The first cohort starts at t=0; subsequent cohorts start at
    multiples of rotation_period_s.
    """
    out: list[dict] = []
    pk_counter = start_pk
    rate = rate_hz
    n_cohorts = int(duration_s // rotation_period_s) + 1
    for cohort in range(n_cohorts):
        cohort_start = cohort * rotation_period_s
        cohort_end = min(duration_s, cohort_start + rotation_period_s)
        if cohort_start >= duration_s:
            break
        cohort_frames = int((cohort_end - cohort_start) * rate)
        for s_off in range(n_concurrent):
            pk = pk_counter
            pk_counter += 1
            x0 = float(s_off * 5.0 + cohort * 100.0)
            for i in range(cohort_frames):
                t = cohort_start + i / rate
                out.append(_msg(
                    pk=pk, sender=pk, t=t, cls=attack_class,
                    posx=x0 + i * 0.5, posy=0.0,
                    spdx=1.0, spdy=0.0,
                ))
    return out


def cold_start_gaming(
    n_rotations: int = 20,
    msgs_per_pseudonym: int = 24,
    rate_hz: float = 10.0,
    start_pk_base: int = 70000,
    start_t: float = 0.0,
    attack_class: int = 5,
) -> list[dict]:
    """One attacker rotates pseudonym below the w/2 cold-start gate. Spec §5.4.

    Default 24 messages < default w/2 = 25, so no single pseudonym ever
    accrues enough observations to leave cold-start. Tests whether the
    framework's revocation gate suppresses hints (it should, by P7), and
    whether per-message classification can still flag the attack stream.
    """
    out: list[dict] = []
    t = start_t
    for r in range(n_rotations):
        pk = start_pk_base + r
        for i in range(msgs_per_pseudonym):
            out.append(_msg(
                pk=pk, sender=10001, t=t, cls=attack_class,
                posx=300.0 + i * 0.5, posy=0.0,
                spdx=1.0, spdy=0.0,
            ))
            t += 1.0 / rate_hz
    return out


# ---------------- scenario builder ----------------

@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    rows: list[dict]
    adversary_pseudonyms: tuple[int, ...]


def merge_streams(*streams: Iterable[dict]) -> list[dict]:
    """Merge multiple message streams in global sendTime order."""
    combined: list[dict] = []
    for s in streams:
        combined.extend(s)
    combined.sort(key=lambda r: r["sendTime"])
    return combined
