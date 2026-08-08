"""Physics-aware adversary constructions.

Defines the A-PHY-1..6 attack classes from the physics-aware threat
model (`framework/specs/physics_aware_threat_model.md`). Each
constructor rewrites VeReMi attack messages to satisfy one or more
of the defender's physics priors:

- A-PHY-1 (`make_kinematic_respecting`): zero kinematic residual e_v.
- A-PHY-2 (`make_kinematic_capped`): + cap implied speed at v_max.
- A-PHY-3 (`make_distribution_matching`): + sample speeds from a
  benign GMM (i.i.d.).
- A-PHY-4 (`make_distribution_matching_persistent`): + per-vehicle
  consistent GMM component (driving style).
- A-PHY-5 (`make_distribution_matching_ar1`): + AR(1) speed
  auto-correlation.
- A-PHY-6 (`make_road_grid_aware`): + snap positions to a coarse
  grid (proxy for road-network constraint).

All constructors share the kinematic-respecting + benign-distribution
sampling base. A-PHY-6 adds a STRUCTURAL constraint that the prior
A-PHY classes did not address.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_kinematic_respecting(
    df: pd.DataFrame,
    target_classes: tuple[int, ...] | None = None,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
) -> pd.DataFrame:
    """A-PHY-1: rewrite attack messages so the reported speed matches
    the implied speed from consecutive positions, eliminating the
    residual e_v signal.
    """
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)

    out = df.copy().reset_index(drop=True)
    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue
        if len(rows) < 2:
            continue

        posx = rows["posx"].values.astype(np.float64)
        posy = rows["posy"].values.astype(np.float64)
        st = rows["sendTime"].values.astype(np.float64)

        dt = np.diff(st)
        dt = np.clip(dt, min_dt, max_dt)
        v_implied_x = np.diff(posx) / dt
        v_implied_y = np.diff(posy) / dt
        v_implied_x = np.concatenate([[posx[1] - posx[0]], v_implied_x])
        v_implied_y = np.concatenate([[posy[1] - posy[0]], v_implied_y])

        a_implied_x = np.zeros_like(posx)
        a_implied_y = np.zeros_like(posx)
        if len(posx) >= 3:
            denom = (((st[2:] - st[1:-1]) + (st[1:-1] - st[:-2])) / 2.0) ** 2
            a_implied_x[1:-1] = (posx[2:] - 2 * posx[1:-1] + posx[:-2]) / denom
            a_implied_y[1:-1] = (posy[2:] - 2 * posy[1:-1] + posy[:-2]) / denom

        local_mask = attack_mask.copy()
        local_mask[0] = False
        for j in range(len(ordered_idx)):
            if not local_mask[j]:
                continue
            i = ordered_idx[j]
            out.at[i, "spdx"] = float(v_implied_x[j])
            out.at[i, "spdy"] = float(v_implied_y[j])
            out.at[i, "aclx"] = float(a_implied_x[j])
            out.at[i, "acly"] = float(a_implied_y[j])

    return out


def fit_benign_speed_gmm(
    benign_df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 17,
):
    """Fit a Gaussian Mixture Model on benign (spdx, spdy, aclx, acly)."""
    from sklearn.mixture import GaussianMixture
    if "class" in benign_df.columns:
        benign = benign_df[benign_df["class"] == 0]
    else:
        benign = benign_df
    feats = benign[["spdx", "spdy", "aclx", "acly"]].to_numpy(dtype=np.float64)
    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state,
        covariance_type="full", max_iter=200, reg_covar=1e-3,
    )
    gmm.fit(feats)
    return gmm


def make_distribution_matching(
    df: pd.DataFrame,
    benign_gmm,
    target_classes: tuple[int, ...] | None = None,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
    seed: int = 17,
) -> pd.DataFrame:
    """A-PHY-3: rewrite attack messages with GMM-sampled benign-looking
    speed trajectories. See module docstring for details.
    """
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)

    out = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue

        st = rows["sendTime"].values.astype(np.float64)
        posx = rows["posx"].values.astype(np.float64).copy()
        posy = rows["posy"].values.astype(np.float64).copy()

        local_rng = np.random.default_rng(seed + int(ordered_idx[0]))
        n = len(ordered_idx)
        samples, _ = benign_gmm.sample(n)
        local_rng.shuffle(samples)
        sampled_spdx = samples[:, 0]
        sampled_spdy = samples[:, 1]
        sampled_aclx = samples[:, 2]
        sampled_acly = samples[:, 3]

        for j in range(1, n):
            if not attack_mask[j]:
                continue
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            posx[j] = posx[j - 1] + sampled_spdx[j] * dt
            posy[j] = posy[j - 1] + sampled_spdy[j] * dt

        for j in range(n):
            if not attack_mask[j] or j == 0:
                continue
            i = ordered_idx[j]
            out.at[i, "posx"] = float(posx[j])
            out.at[i, "posy"] = float(posy[j])
            out.at[i, "spdx"] = float(sampled_spdx[j])
            out.at[i, "spdy"] = float(sampled_spdy[j])
            out.at[i, "aclx"] = float(sampled_aclx[j])
            out.at[i, "acly"] = float(sampled_acly[j])

    return out


def make_distribution_matching_persistent(
    df: pd.DataFrame,
    benign_gmm,
    target_classes: tuple[int, ...] | None = None,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
    seed: int = 17,
) -> pd.DataFrame:
    """A-PHY-4: per-vehicle persistent-component GMM sampling."""
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)

    out = df.copy().reset_index(drop=True)
    n_components = benign_gmm.n_components
    mixture_weights = benign_gmm.weights_
    means = benign_gmm.means_
    covs = benign_gmm.covariances_

    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue

        local_rng = np.random.default_rng(seed + int(ordered_idx[0]))
        comp = local_rng.choice(n_components, p=mixture_weights)
        comp_mean = means[comp]
        comp_cov = covs[comp]

        st = rows["sendTime"].values.astype(np.float64)
        posx = rows["posx"].values.astype(np.float64).copy()
        posy = rows["posy"].values.astype(np.float64).copy()

        n = len(ordered_idx)
        sampled = local_rng.multivariate_normal(comp_mean, comp_cov, size=n)
        s_spdx = sampled[:, 0]
        s_spdy = sampled[:, 1]
        s_aclx = sampled[:, 2]
        s_acly = sampled[:, 3]

        for j in range(1, n):
            if not attack_mask[j]:
                continue
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            posx[j] = posx[j - 1] + s_spdx[j] * dt
            posy[j] = posy[j - 1] + s_spdy[j] * dt

        for j in range(n):
            if not attack_mask[j] or j == 0:
                continue
            i = ordered_idx[j]
            out.at[i, "posx"] = float(posx[j])
            out.at[i, "posy"] = float(posy[j])
            out.at[i, "spdx"] = float(s_spdx[j])
            out.at[i, "spdy"] = float(s_spdy[j])
            out.at[i, "aclx"] = float(s_aclx[j])
            out.at[i, "acly"] = float(s_acly[j])

    return out


def make_distribution_matching_ar1(
    df: pd.DataFrame,
    benign_gmm,
    alpha: float = 0.9,
    target_classes: tuple[int, ...] | None = None,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
    seed: int = 17,
) -> pd.DataFrame:
    """A-PHY-5: AR(1)-smoothed GMM speed sampling."""
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)
    alpha = float(max(0.0, min(1.0, alpha)))

    out = df.copy().reset_index(drop=True)
    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue

        st = rows["sendTime"].values.astype(np.float64)
        posx = rows["posx"].values.astype(np.float64).copy()
        posy = rows["posy"].values.astype(np.float64).copy()

        n = len(ordered_idx)
        raw, _ = benign_gmm.sample(n)
        local_rng = np.random.default_rng(seed + int(ordered_idx[0]))
        local_rng.shuffle(raw)
        smoothed = np.zeros_like(raw)
        smoothed[0] = raw[0]
        for j in range(1, n):
            smoothed[j] = alpha * smoothed[j - 1] + (1.0 - alpha) * raw[j]

        s_spdx = smoothed[:, 0]
        s_spdy = smoothed[:, 1]
        s_aclx = smoothed[:, 2]
        s_acly = smoothed[:, 3]

        for j in range(1, n):
            if not attack_mask[j]:
                continue
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            posx[j] = posx[j - 1] + s_spdx[j] * dt
            posy[j] = posy[j - 1] + s_spdy[j] * dt

        for j in range(n):
            if not attack_mask[j] or j == 0:
                continue
            i = ordered_idx[j]
            out.at[i, "posx"] = float(posx[j])
            out.at[i, "posy"] = float(posy[j])
            out.at[i, "spdx"] = float(s_spdx[j])
            out.at[i, "spdy"] = float(s_spdy[j])
            out.at[i, "aclx"] = float(s_aclx[j])
            out.at[i, "acly"] = float(s_acly[j])

    return out


def make_road_grid_aware(
    df: pd.DataFrame,
    benign_gmm,
    grid_size: float = 50.0,
    alpha: float = 0.9,
    target_classes: tuple[int, ...] | None = None,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
    seed: int = 17,
) -> pd.DataFrame:
    """A-PHY-6: road-grid-snapped + AR(1) GMM trajectory.

    Extends A-PHY-5 by additionally snapping positions to a coarse
    grid (default 50m squares) - a proxy for road-network constraint
    that does not require an actual OSM map. Trajectories therefore
    move along grid axes rather than random-walk in continuous space.

    Implementation:
    - At each attack message, compute the AR(1)-smoothed GMM speed
      sample (as A-PHY-5).
    - Integrate position forward.
    - Snap the integrated position to the nearest grid intersection
      every K steps (K = grid_size / typical_step_size).

    The recomputed spdx, spdy are derived from the SNAPPED positions
    so kinematic consistency is preserved.
    """
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)
    alpha = float(max(0.0, min(1.0, alpha)))

    out = df.copy().reset_index(drop=True)
    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue

        st = rows["sendTime"].values.astype(np.float64)
        posx = rows["posx"].values.astype(np.float64).copy()
        posy = rows["posy"].values.astype(np.float64).copy()

        n = len(ordered_idx)
        raw, _ = benign_gmm.sample(n)
        local_rng = np.random.default_rng(seed + int(ordered_idx[0]))
        local_rng.shuffle(raw)
        smoothed = np.zeros_like(raw)
        smoothed[0] = raw[0]
        for j in range(1, n):
            smoothed[j] = alpha * smoothed[j - 1] + (1.0 - alpha) * raw[j]
        s_spdx = smoothed[:, 0]
        s_spdy = smoothed[:, 1]

        # Integrate positions, then snap to grid every step.
        new_posx = np.zeros_like(posx)
        new_posy = np.zeros_like(posy)
        new_posx[0] = posx[0]
        new_posy[0] = posy[0]
        for j in range(1, n):
            if not attack_mask[j]:
                new_posx[j] = posx[j]
                new_posy[j] = posy[j]
                continue
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            raw_x = new_posx[j - 1] + s_spdx[j] * dt
            raw_y = new_posy[j - 1] + s_spdy[j] * dt
            # Snap to grid.
            new_posx[j] = round(raw_x / grid_size) * grid_size
            new_posy[j] = round(raw_y / grid_size) * grid_size

        # Re-derive speeds from snapped positions.
        new_spdx = np.zeros_like(new_posx)
        new_spdy = np.zeros_like(new_posy)
        new_aclx = np.zeros_like(new_posx)
        new_acly = np.zeros_like(new_posy)
        for j in range(1, n):
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            new_spdx[j] = (new_posx[j] - new_posx[j - 1]) / dt
            new_spdy[j] = (new_posy[j] - new_posy[j - 1]) / dt
        for j in range(2, n):
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            new_aclx[j] = (new_spdx[j] - new_spdx[j - 1]) / dt
            new_acly[j] = (new_spdy[j] - new_spdy[j - 1]) / dt

        for j in range(n):
            if not attack_mask[j] or j == 0:
                continue
            i = ordered_idx[j]
            out.at[i, "posx"] = float(new_posx[j])
            out.at[i, "posy"] = float(new_posy[j])
            out.at[i, "spdx"] = float(new_spdx[j])
            out.at[i, "spdy"] = float(new_spdy[j])
            out.at[i, "aclx"] = float(new_aclx[j])
            out.at[i, "acly"] = float(new_acly[j])

    return out


def make_kinematic_capped(
    df: pd.DataFrame,
    target_classes: tuple[int, ...] | None = None,
    v_max: float = 30.0,
    min_dt: float = 0.05,
    max_dt: float = 1.0,
) -> pd.DataFrame:
    """A-PHY-2: rewrite attack messages to satisfy kinematic continuity
    AND magnitude bounds.
    """
    if "class" not in df.columns:
        raise ValueError("df must have 'class' column")
    if target_classes is None:
        target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
    else:
        target_classes_set = set(int(c) for c in target_classes)

    out = df.copy().reset_index(drop=True)
    group_cols = ["senderPseudo"] if "senderPseudo" in out.columns else ["sender"]
    for _, idx in out.groupby(group_cols, observed=True, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = out.loc[idx]
        order = np.argsort(sub["sendTime"].values, kind="stable")
        ordered_idx = idx[order]
        rows = out.loc[ordered_idx]

        cls = rows["class"].values.astype(int)
        attack_mask = np.array([int(c) in target_classes_set for c in cls])
        if not attack_mask.any():
            continue

        posx = rows["posx"].values.astype(np.float64).copy()
        posy = rows["posy"].values.astype(np.float64).copy()
        st = rows["sendTime"].values.astype(np.float64)

        for j in range(1, len(ordered_idx)):
            if not attack_mask[j]:
                continue
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            dev_x = posx[j] - posx[j - 1]
            dev_y = posy[j] - posy[j - 1]
            dev_norm = (dev_x ** 2 + dev_y ** 2) ** 0.5
            cap = v_max * dt
            if dev_norm > cap and dev_norm > 0:
                scale = cap / dev_norm
                posx[j] = posx[j - 1] + dev_x * scale
                posy[j] = posy[j - 1] + dev_y * scale

        v_x = np.zeros_like(posx)
        v_y = np.zeros_like(posy)
        for j in range(1, len(ordered_idx)):
            dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
            v_x[j] = (posx[j] - posx[j - 1]) / dt
            v_y[j] = (posy[j] - posy[j - 1]) / dt
        v_x[0] = v_x[1] if len(v_x) > 1 else 0.0
        v_y[0] = v_y[1] if len(v_y) > 1 else 0.0

        local_mask = attack_mask.copy()
        local_mask[0] = False
        for j in range(len(ordered_idx)):
            if not local_mask[j]:
                continue
            i = ordered_idx[j]
            out.at[i, "posx"] = float(posx[j])
            out.at[i, "posy"] = float(posy[j])
            out.at[i, "spdx"] = float(v_x[j])
            out.at[i, "spdy"] = float(v_y[j])
            if j >= 2:
                dt = max(min_dt, min(max_dt, st[j] - st[j - 1]))
                a_x = (v_x[j] - v_x[j - 1]) / dt
                a_y = (v_y[j] - v_y[j - 1]) / dt
                out.at[i, "aclx"] = float(a_x)
                out.at[i, "acly"] = float(a_y)

    return out


@dataclass(frozen=True)
class AttackRewriteSummary:
    """Summary of a rewrite operation for logging."""
    n_total: int
    n_attack: int
    n_rewritten: int
    n_unchanged_attack: int

    def short(self) -> str:
        return (
            f"rows={self.n_total}  attack={self.n_attack}  "
            f"rewritten={self.n_rewritten}  unchanged_attack={self.n_unchanged_attack}"
        )


def rewrite_with_summary(
    df: pd.DataFrame,
    target_classes: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, AttackRewriteSummary]:
    """Wrapper around make_kinematic_respecting returning a summary."""
    n_total = len(df)
    if "class" in df.columns:
        if target_classes is None:
            target_classes_set = set(int(c) for c in df["class"].unique() if c != 0)
        else:
            target_classes_set = set(int(c) for c in target_classes)
        attack_mask = df["class"].isin(target_classes_set)
        n_attack = int(attack_mask.sum())
    else:
        n_attack = 0
    new_df = make_kinematic_respecting(df, target_classes=target_classes)
    if n_attack > 0 and "spdx" in df.columns:
        delta = np.abs(new_df["spdx"].values - df["spdx"].values) + \
                np.abs(new_df["spdy"].values - df["spdy"].values)
        n_rewritten = int((delta > 1e-9).sum())
    else:
        n_rewritten = 0
    n_unchanged_attack = n_attack - n_rewritten
    return new_df, AttackRewriteSummary(
        n_total=n_total, n_attack=n_attack,
        n_rewritten=n_rewritten, n_unchanged_attack=n_unchanged_attack,
    )
