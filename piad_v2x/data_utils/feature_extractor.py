"""Feature extractor.

Computes per-message features against the flattened
VeReMi Extension CSV.

For each message m_t the extractor produces:
- Raw kinematic fields (noisy variants `_n`, matching what a receiver sees).
- Kinematic deltas vs the prior message from the same pseudonym
  (Δposx, Δposy, Δspd, Δhed, Δτ).
- Inferred motion: implied speed v*, implied acceleration a*, turn rate r*.
- Consistency residuals: e_v = v* - v_reported, e_a = a* - a_reported.
- Cold-start flag: True when no prior message from the same pseudonym exists
  (the kinematic-residual features are masked to 0 for these rows).
- Physics-plausibility flags (S1, VCADS-style hard-bound violation margins,
  each in [0, 1], higher = less plausible): speed-limit, accel-limit,
  position-jump (implied-speed limit), speed/position consistency, and
  heading/motion consistency. The instantaneous flags (speed, accel) are
  valid on every row; the delta-dependent ones are masked to 0 on cold-start.
  These are the feature-level physics-plausibility flags.

Output is a pure numerical feature matrix plus label vector. Group identifiers
(sender, senderPseudo) are returned separately for downstream use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Order of the feature vector x_t. Stable across calls; used to label columns.
FEATURE_NAMES = (
    # Raw fields (8)
    "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy",
    # Kinematic deltas (5)
    "d_posx", "d_posy", "d_spd_mag", "d_hed_mag", "d_t",
    # Inferred motion (3)
    "v_star", "a_star", "r_star",
    # Consistency residuals (2)
    "e_v", "e_a",
    # Cold-start mask (1)
    "cold",
    # Physics-plausibility flags (5) - VCADS-style [0,1] violation margins
    "p_spd_lim", "p_acl_lim", "p_posjump", "p_spd_cons", "p_hed_cons",
)


@dataclass(frozen=True)
class FeatureConfig:
    # Δτ guards. Two consecutive messages with effectively zero Δτ would
    # produce divergent v*, a*; we clip the divisor and mask via cold flag.
    min_dt_seconds: float = 1e-6
    # Maximum plausible Δτ between consecutive same-pseudonym messages
    # (10 Hz nominal; allow gaps up to 5 s before treating as cold-start).
    max_dt_seconds: float = 5.0

    # S1 physics-plausibility bounds (VCADS-style). Generous so a flag firing
    # means a clear physical violation, not normal driving.
    v_max_mps: float = 70.0          # ~252 km/h upper bound for road vehicles
    a_max_mps2: float = 10.0         # ~1g longitudinal/lateral envelope
    tau_v_consistency: float = 5.0   # |v* - v_reported| tolerance band (m/s)
    min_speed_for_heading: float = 1.0  # below this, heading is ill-defined


@dataclass(frozen=True)
class FeatureBatch:
    X: np.ndarray                  # shape (n, len(FEATURE_NAMES))
    y: np.ndarray                  # shape (n,)
    senders: np.ndarray            # shape (n,) - for grouped CV / inspection
    pseudonyms: np.ndarray         # shape (n,)
    feature_names: tuple = field(default=FEATURE_NAMES)


def extract_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> FeatureBatch:
    """Compute features from a sorted VeReMi DataFrame.

    The input must be sorted by (senderPseudo, sendTime) so within-group shifts
    pick up the correct prior message. `dataset.load_messages` does this.
    """
    cfg = config or FeatureConfig()

    # Use the simulator-truth columns; their physical scales are correct
    # (position in m, speed in m/s, acceleration in m/s², heading components).
    # The `_n` noise-delta columns are retained in the loaded frame for a
    # future noise-composition variant. See dataset.py NOISE_DELTA_COLS.
    posx = df["posx"].to_numpy(dtype=float)
    posy = df["posy"].to_numpy(dtype=float)
    spdx = df["spdx"].to_numpy(dtype=float)
    spdy = df["spdy"].to_numpy(dtype=float)
    aclx = df["aclx"].to_numpy(dtype=float)
    acly = df["acly"].to_numpy(dtype=float)
    hedx = df["hedx"].to_numpy(dtype=float)
    hedy = df["hedy"].to_numpy(dtype=float)
    t = df["sendTime"].to_numpy(dtype=float)
    pseudo = df["senderPseudo"].to_numpy()
    sender = df["sender"].to_numpy()
    y = df["class"].to_numpy()

    # Same-pseudonym mask: True for rows whose previous row shares senderPseudo.
    same_pk = np.zeros(len(df), dtype=bool)
    same_pk[1:] = pseudo[1:] == pseudo[:-1]

    # Within-group deltas (only meaningful where same_pk is True).
    d_t = np.zeros_like(t)
    d_posx = np.zeros_like(posx)
    d_posy = np.zeros_like(posy)
    d_spdx = np.zeros_like(spdx)
    d_spdy = np.zeros_like(spdy)
    d_hedx = np.zeros_like(hedx)
    d_hedy = np.zeros_like(hedy)
    d_t[1:] = t[1:] - t[:-1]
    d_posx[1:] = posx[1:] - posx[:-1]
    d_posy[1:] = posy[1:] - posy[:-1]
    d_spdx[1:] = spdx[1:] - spdx[:-1]
    d_spdy[1:] = spdy[1:] - spdy[:-1]
    d_hedx[1:] = hedx[1:] - hedx[:-1]
    d_hedy[1:] = hedy[1:] - hedy[:-1]

    # Cold-start: not same-pseudonym, or Δτ outside the plausible window.
    cold_flag = ~same_pk | (d_t <= cfg.min_dt_seconds) | (d_t > cfg.max_dt_seconds)

    # Safe divisor: clip Δτ; cold rows are masked at the end anyway.
    safe_dt = np.clip(d_t, cfg.min_dt_seconds, None)

    # Magnitudes (rotation-invariant deltas).
    d_spd_mag = np.hypot(d_spdx, d_spdy)
    d_hed_mag = np.hypot(d_hedx, d_hedy)

    # Inferred motion.
    v_star = np.hypot(d_posx, d_posy) / safe_dt
    spd_reported = np.hypot(spdx, spdy)
    acl_reported = np.hypot(aclx, acly)
    a_star = d_spd_mag / safe_dt
    r_star = d_hed_mag / safe_dt

    # Consistency residuals.
    e_v = v_star - spd_reported
    e_a = a_star - acl_reported

    # Mask kinematic-residual features on cold-start rows.
    for arr in (d_posx, d_posy, d_spd_mag, d_hed_mag, d_t, v_star, a_star, r_star, e_v, e_a):
        arr[cold_flag] = 0.0

    # --- S1: physics-plausibility flags (VCADS-style), each in [0, 1]. ---
    # Soft hinge: 0 inside the bound, ramping to 1 at twice the bound.
    def _margin(x: np.ndarray, bound: float) -> np.ndarray:
        return np.clip(np.abs(x) / bound - 1.0, 0.0, 1.0)

    # Instantaneous flags (reported fields only; valid on every row).
    p_spd_lim = _margin(spd_reported, cfg.v_max_mps)
    p_acl_lim = _margin(acl_reported, cfg.a_max_mps2)
    # Delta-dependent flags. v_star / e_v are already 0 on cold rows, so these
    # are naturally 0 there; the heading flag is guarded and cold-masked below.
    p_posjump = _margin(v_star, cfg.v_max_mps)
    p_spd_cons = _margin(e_v, cfg.tau_v_consistency)
    # Heading vs motion-direction mismatch in [0, 1] = (1 - cos)/2.
    motion_norm = np.hypot(d_posx, d_posy)
    hed_norm = np.hypot(hedx, hedy)
    valid_hed = (~cold_flag) & (motion_norm > cfg.min_dt_seconds) \
        & (hed_norm > cfg.min_dt_seconds) & (v_star > cfg.min_speed_for_heading)
    safe_mn = np.where(motion_norm > 0, motion_norm, 1.0)
    safe_hn = np.where(hed_norm > 0, hed_norm, 1.0)
    cos_sim = (d_posx * hedx + d_posy * hedy) / (safe_mn * safe_hn)
    p_hed_cons = np.where(valid_hed, np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0), 0.0)

    X = np.column_stack([
        posx, posy, spdx, spdy, aclx, acly, hedx, hedy,
        d_posx, d_posy, d_spd_mag, d_hed_mag, d_t,
        v_star, a_star, r_star,
        e_v, e_a,
        cold_flag.astype(float),
        p_spd_lim, p_acl_lim, p_posjump, p_spd_cons, p_hed_cons,
    ])

    return FeatureBatch(
        X=X,
        y=y.astype(int),
        senders=sender,
        pseudonyms=pseudo,
    )


def extract_features_to_dataframe(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Same as `extract_features` but returns a DataFrame for inspection.

    Slower; use `extract_features` for training pipelines.
    """
    batch = extract_features(df, config)
    out = pd.DataFrame(batch.X, columns=list(FEATURE_NAMES))
    out["class"] = batch.y
    out["sender"] = batch.senders
    out["senderPseudo"] = batch.pseudonyms
    return out
