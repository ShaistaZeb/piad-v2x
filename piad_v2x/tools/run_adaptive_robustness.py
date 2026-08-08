"""Adaptive-robustness evaluation: multi-scale detector vs a goal-preserving white-box attacker.

Runs the confirmatory adaptive-robustness protocol. Run
from the repository root (after `pip install -e .`):

    python piad_v2x/tools/run_adaptive_robustness.py --out experiments/results/adaptive_robustness.json

WHAT CHANGED (the fix that makes this valid). The earlier pilot lacked the type-4
ground-truth trajectory, so its attacker could only jitter around an existing
falsification and could not be stopped from silently abandoning the attack. The
extractor now joins the true transmitted state (`true_px/true_py/true_sx/true_sy`,
`has_truth`) onto every message. The attacker is therefore GOAL-PRESERVING:

  goal:  the perturbed reported position must stay at least  GOAL_FRAC * |report0 - true|
         away from the vehicle's TRUE position, so the spoof is not abandoned.

Within that goal, the attacker minimises the detector's misbehaviour score. Its
strongest move is not random jitter but CONSISTENCY-SEEKING: report the
constant-velocity prediction from its own previous message (drives the single-vehicle
residuals r_pos_cv / r_spd_pos toward zero) and then push the point radially away from
the true position back to the goal boundary. eps interpolates from "no move" (eps=0, the
naive VeReMi attacker, harness-correctness gate) to "full consistency-seeking + full
random search radius" (eps=1). Random candidates supplement the consistency candidate.

  * Physical realisability (hard): speed <= V_MAX; the reported speed is set to the
    displacement-implied speed so displacement/dt is self-consistent by construction.
  This is a WITHIN-DETECTOR SCALE ABLATION of one multi-scale physics detector, not a
  contest between rival detectors: "single-vehicle" = the kinematic-scale-only ablation
  (KINEMATIC features); "relational" = the full multi-scale detector (KINEMATIC + the
  multi-vehicle interaction invariants). It localises which physical scale carries adaptive
  robustness.

  * Symmetric attacker: the SAME perturbed scene is scored by both the kinematic-scale-only
    ablation and the full multi-scale detector, so any difference in degradation is
    attributable to the multi-vehicle scale, not to a weaker attack on one arm.
  * `--target single` (default): the attacker optimises the kinematic surface it directly
    controls; the full multi-scale detector is then scored on the resulting scene.
    This is the primary run.
  * `--target relational`: the attacker instead optimises against the multi-vehicle scale
    itself (full-scene interaction recompute per candidate set). Heavier; run on the core
    scenario to close pre-reg amendment 2 (does the multi-vehicle robustness survive an
    attacker that targets IT, not just its own kinematics?). Non-colluding: the attacker still moves
    only its own messages; neighbours are fixed.

Cluster bootstrap over senders gives the 95% CI on the eps=1.0 gap (pre-reg section 6).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from piad_v2x.models.relational import relational_features, KINEMATIC, REL_TRUE

BREAK_AUC = 0.60
V_MAX = 50.0            # m/s physical realisability cap (matches relational.py VMAX)
SEED = 42
GOAL_FRAC = 0.8        # attacker must retain >=80% of its original spoof magnitude
RADIUS_POS = 25.0      # metres std of the supplementary random search at eps = 1.0
N_BOOT = 1000


def recompute_single_vehicle(df, px, py, sx, sy):
    """Return a copy of df with (px,py,sx,sy) replaced and the derived single-vehicle
    features recomputed from each row's own previous report. df must already carry
    prev_px/prev_py/prev_sx/prev_sy/dt/has_prev."""
    out = df.copy()
    out["px"], out["py"], out["sx"], out["sy"] = px, py, sx, sy
    out["spd_mag"] = np.hypot(sx, sy)
    dt = out["dt"].to_numpy()
    dt = np.where(dt <= 0, 1.0, dt)
    hp = out["has_prev"].to_numpy().astype(bool)
    ppx, ppy = out["prev_px"].to_numpy(), out["prev_py"].to_numpy()
    psx, psy = out["prev_sx"].to_numpy(), out["prev_sy"].to_numpy()
    pred_px, pred_py = ppx + psx * dt, ppy + psy * dt
    r_pos_cv = np.hypot(px - pred_px, py - pred_py)
    dispx, dispy = px - ppx, py - ppy
    disp = np.hypot(dispx, dispy)
    impl_sx, impl_sy = dispx / dt, dispy / dt
    r_spd_pos = np.hypot(sx - impl_sx, sy - impl_sy)
    implied = np.hypot(impl_sx, impl_sy)
    out["disp"] = np.where(hp, disp, 0.0)
    out["implied_spd"] = np.where(hp, implied, 0.0)
    out["r_pos_cv"] = np.where(hp, r_pos_cv, 0.0)
    out["r_spd_pos"] = np.where(hp, r_spd_pos, 0.0)
    return out


def _goal_project(cpx, cpy, tpx, tpy, off_hat_x, off_hat_y, goal_min):
    """Push each candidate position out to >= goal_min from its true position, so the
    attack is never abandoned. If the candidate collapses onto true, push along the
    original spoof direction (off_hat)."""
    vx, vy = cpx - tpx, cpy - tpy
    d = np.hypot(vx, vy)
    safe = d > 1e-6
    scale = np.where((d < goal_min) & safe, goal_min / np.maximum(d, 1e-9), 1.0)
    ox = np.where(safe, vx * scale, off_hat_x * goal_min)
    oy = np.where(safe, vy * scale, off_hat_y * goal_min)
    return tpx + ox, tpy + oy


def _implied_speed(cpx, cpy, ppx, ppy, dt):
    sx = (cpx - ppx) / dt
    sy = (cpy - ppy) / dt
    spd = np.hypot(sx, sy)
    scale = np.where(spd > V_MAX, V_MAX / np.maximum(spd, 1e-9), 1.0)
    return sx * scale, sy * scale


def goal_preserving_attack(atk_df, scorer, eps, k_cand, rng):
    """Goal-preserving white-box evasion. `scorer(px,py,sx,sy)->p_mal[n]` scores a
    candidate assignment for the attacker rows against the targeted detector. Returns the
    best perturbed (px,py,sx,sy) per row (lowest score subject to goal + physical caps)."""
    n = len(atk_df)
    px0 = atk_df["px"].to_numpy(float);  py0 = atk_df["py"].to_numpy(float)
    sx0 = atk_df["sx"].to_numpy(float);  sy0 = atk_df["sy"].to_numpy(float)
    ppx = atk_df["prev_px"].to_numpy(float); ppy = atk_df["prev_py"].to_numpy(float)
    psx = atk_df["prev_sx"].to_numpy(float); psy = atk_df["prev_sy"].to_numpy(float)
    tpx = atk_df["true_px"].to_numpy(float); tpy = atk_df["true_py"].to_numpy(float)
    dt = atk_df["dt"].to_numpy(float); dt = np.where(dt <= 0, 1.0, dt)

    off_x, off_y = px0 - tpx, py0 - tpy
    off0 = np.hypot(off_x, off_y)
    off_hat_x = np.where(off0 > 1e-6, off_x / np.maximum(off0, 1e-9), 1.0)
    off_hat_y = np.where(off0 > 1e-6, off_y / np.maximum(off0, 1e-9), 0.0)
    goal_min = GOAL_FRAC * off0

    best_px, best_py, best_sx, best_sy = px0.copy(), py0.copy(), sx0.copy(), sy0.copy()
    best_p = scorer(best_px, best_py, best_sx, best_sy)
    if eps == 0.0:
        return best_px, best_py, best_sx, best_sy

    def consider(cpx, cpy):
        nonlocal best_px, best_py, best_sx, best_sy, best_p
        cpx, cpy = _goal_project(cpx, cpy, tpx, tpy, off_hat_x, off_hat_y, goal_min)
        csx, csy = _implied_speed(cpx, cpy, ppx, ppy, dt)
        p = scorer(cpx, cpy, csx, csy)
        take = p < best_p
        best_p = np.where(take, p, best_p)
        best_px = np.where(take, cpx, best_px); best_py = np.where(take, cpy, best_py)
        best_sx = np.where(take, csx, best_sx); best_sy = np.where(take, csy, best_sy)

    # (1) consistency-seeking candidate: CV prediction from own previous report,
    #     moved a fraction eps of the way there. Drives single-vehicle residuals down.
    cv_px = ppx + psx * dt
    cv_py = ppy + psy * dt
    consider(px0 + eps * (cv_px - px0), py0 + eps * (cv_py - py0))

    # (2) supplementary random search within an eps-scaled radius around the report
    r = eps * RADIUS_POS
    for _ in range(k_cand):
        consider(px0 + rng.normal(0, r, n), py0 + rng.normal(0, r, n))
    return best_px, best_py, best_sx, best_sy


def add_prev_columns(df):
    """Attach each message's previous reported (px,py,sx,sy) within its
    (receiver, senderPseudo) sequence, for exact residual recomputation."""
    df = df.sort_values(["receiver", "senderPseudo", "rcvTime"], kind="stable").reset_index(drop=True)
    g = df.groupby(["receiver", "senderPseudo"], sort=False)
    for c in ("px", "py", "sx", "sy"):
        df["prev_" + c] = g[c].shift(1).fillna(df[c])
    return df


def _cluster_bootstrap_gap(yte, groups_te, p_single, p_rel, rng):
    """95% CI on (AUC_rel - AUC_single) resampling senders (clusters)."""
    ug = np.unique(groups_te)
    gaps = []
    for _ in range(N_BOOT):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(groups_te == g)[0] for g in pick])
        yb = yte[idx]
        if yb.min() == yb.max():
            continue
        gaps.append(roc_auc_score(yb, p_rel[idx]) - roc_auc_score(yb, p_single[idx]))
    if not gaps:
        return (float("nan"), float("nan"))
    return (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))


def evaluate_scenario(path, eps_grid, k_cand, max_attackers, target, rng):
    scen = os.path.basename(path).replace(".parquet", "")
    df = pd.read_parquet(path)
    df = df[df.attackerType >= 0].reset_index(drop=True)
    if "true_px" not in df.columns or df["has_truth"].mean() < 0.5:
        raise SystemExit(f"{scen}: ground truth missing/sparse; re-extract with the "
                         "updated veremi_extract.py before running adaptive robustness.")
    df = add_prev_columns(df)
    rel = relational_features(df)
    df = df.merge(rel, on="messageID", how="left")
    y = df["is_attacker"].to_numpy()
    gid = df["sender"].to_numpy()
    # replicate relational.py's split EXACTLY so eps=0 reproduces the committed AUC
    split_rng = np.random.default_rng(42)
    ug = np.unique(gid); split_rng.shuffle(ug)
    te = set(ug[:int(0.3 * len(ug))])
    mask_te = np.array([g in te for g in gid])
    yte = y[mask_te]; groups_te = gid[mask_te]

    def fit(cols):
        m = RandomForestClassifier(n_estimators=120, max_depth=16, n_jobs=-1,
                                   class_weight="balanced", random_state=42)
        m.fit(df.loc[~mask_te, cols].fillna(0).values, y[~mask_te])
        return m
    rf_single = fit(KINEMATIC)
    rf_rel = fit(KINEMATIC + REL_TRUE)

    atk_idx = np.where(mask_te & (y == 1))[0]
    if len(atk_idx) > max_attackers:
        atk_idx = rng.choice(atk_idx, size=max_attackers, replace=False)
    atk_idx = np.sort(atk_idx)

    curve = {"single_vehicle": {}, "relational": {}, "eps1_gap_ci95": None,
             "n_attacked": int(len(atk_idx)), "target": target,
             "goal_frac": GOAL_FRAC}

    for eps in eps_grid:
        work = df.copy()
        if eps > 0.0:
            atk_df = df.iloc[atk_idx]

            if target == "single":
                def scorer(px, py, sx, sy, _adf=atk_df):
                    cand = recompute_single_vehicle(_adf, px, py, sx, sy)
                    return rf_single.predict_proba(cand[KINEMATIC].fillna(0).values)[:, 1]
            else:  # relational-targeting: recompute relational on the perturbed scene
                loc = work.index[atk_idx]
                cols_sv = ("spd_mag", "disp", "implied_spd", "r_pos_cv", "r_spd_pos")
                def scorer(px, py, sx, sy, _loc=loc, _adf=atk_df):
                    sc = work.copy()
                    sc.loc[_loc, ["px", "py", "sx", "sy"]] = np.column_stack([px, py, sx, sy])
                    rec = recompute_single_vehicle(_adf, px, py, sx, sy)
                    for c in cols_sv:
                        sc.loc[_loc, c] = rec[c].to_numpy()
                    r2 = relational_features(sc)
                    sc = sc.drop(columns=REL_TRUE, errors="ignore").merge(r2, on="messageID", how="left")
                    return rf_rel.predict_proba(
                        sc.loc[_loc, KINEMATIC + REL_TRUE].fillna(0).values)[:, 1]

            bpx, bpy, bsx, bsy = goal_preserving_attack(atk_df, scorer, eps, k_cand, rng)
            work.loc[work.index[atk_idx], ["px", "py", "sx", "sy"]] = np.column_stack([bpx, bpy, bsx, bsy])
            recomputed = recompute_single_vehicle(work.iloc[atk_idx], bpx, bpy, bsx, bsy)
            for c in ("spd_mag", "disp", "implied_spd", "r_pos_cv", "r_spd_pos"):
                work.iloc[atk_idx, work.columns.get_loc(c)] = recomputed[c].to_numpy()
            rel2 = relational_features(work)
            work = work.drop(columns=REL_TRUE, errors="ignore").merge(rel2, on="messageID", how="left")

        p_single = rf_single.predict_proba(work.loc[mask_te, KINEMATIC].fillna(0).values)[:, 1]
        p_rel = rf_rel.predict_proba(work.loc[mask_te, KINEMATIC + REL_TRUE].fillna(0).values)[:, 1]
        a_single = roc_auc_score(yte, p_single)
        a_rel = roc_auc_score(yte, p_rel)
        curve["single_vehicle"][str(eps)] = round(float(a_single), 4)
        curve["relational"][str(eps)] = round(float(a_rel), 4)
        if abs(eps - 1.0) < 1e-9:
            lo, hi = _cluster_bootstrap_gap(yte, groups_te, p_single, p_rel, rng)
            curve["eps1_gap_ci95"] = [round(lo, 4), round(hi, 4)]
        print(f"    [{scen}] eps={eps:.2f}  single={a_single:.4f}  relational={a_rel:.4f}", flush=True)
    # integrals (trapezoid) over the eps grid (np.trapezoid on numpy>=2, else np.trapz)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    xs = [float(e) for e in eps_grid]
    curve["auc_integral_single"] = round(float(trapz(
        [curve["single_vehicle"][str(e)] for e in eps_grid], xs)), 4)
    curve["auc_integral_relational"] = round(float(trapz(
        [curve["relational"][str(e)] for e in eps_grid], xs)), 4)
    return scen, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--eps", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--k-cand", type=int, default=16, help="random search candidates per message")
    ap.add_argument("--max-attackers", type=int, default=8000, help="cap on attacked messages")
    ap.add_argument("--target", choices=["single", "relational"], default="single",
                    help="detector surface the attacker optimises against")
    ap.add_argument("--out", default="experiments/results/adaptive_robustness.json")
    args = ap.parse_args()

    paths = args.scenarios or sorted(glob.glob("data/features/*.parquet"))
    if not paths:
        raise SystemExit("No scenario parquet found. Run piad_v2x/tools/veremi_extract.py first.")
    rng = np.random.default_rng(SEED)
    report = {}
    for p in paths:
        scen, curve = evaluate_scenario(p, args.eps, args.k_cand, args.max_attackers,
                                        args.target, rng)
        report[scen] = curve
    summary = {
        "protocol": "adaptive robustness (goal-preserving white-box attacker)",
        "eps_grid": args.eps, "break_auc": BREAK_AUC, "goal_frac": GOAL_FRAC,
        "k_cand": args.k_cand, "max_attackers": args.max_attackers,
        "attacker": "goal-preserving white-box (uses type-4 true trajectory); "
                    f"target={args.target}; consistency-seeking + random search; "
                    "reported position held >= goal_frac*|report0-true| from true.",
        "scenarios": report,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
