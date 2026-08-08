#!/usr/bin/env python3
"""
Adaptive / white-box attacker case study.

Constructs a physics-RESPECTING spoofer on top of REAL benign trajectories: it
adds a constant drift velocity so claimed position diverges from truth over time
while claimed speed is shifted consistently, keeping every motion-model residual
inside the benign envelope. The detector - which checks internal kinematic
consistency - is structurally blind to this. We report the detector's flag rate
on the adaptive attacker vs the naive (real VeReMi) attacker, and the position
offset achieved while remaining unflagged.

INTEGRITY: these adaptive-attacker rows are CONSTRUCTED, not VeReMi ground truth.
They are used only for an adversarial robustness case study and are never mixed
into the training data or reported as benchmark results.
"""
import os, math, json
import numpy as np, pandas as pd, joblib

MODELS = "experiments/results/models"
OUT = "experiments/results/adaptive_attacker.json"
WIN_M = 25.0  # attacker "wins" a message if claimed pos is >25 m from truth


def norm2(x, y): return np.sqrt(x * x + y * y)


def build_features(seq):
    """seq: DataFrame of one (receiver,pseudo) benign sequence sorted by rcvTime,
    with columns px,py,sx,sy,ax,ay,hx,hy,rcvTime. Recompute the extractor features."""
    rows = []
    prev = None
    for _, m in seq.iterrows():
        px, py, sx, sy = m.px, m.py, m.sx, m.sy
        ax, ay, hx, hy = m.ax, m.ay, m.hx, m.hy
        spd_mag = norm2(sx, sy); acl_mag = norm2(ax, ay)
        r = dict(px=px, py=py, sx=sx, sy=sy, ax=ax, ay=ay, hx=hx, hy=hy,
                 spd_mag=spd_mag, acl_mag=acl_mag)
        if prev is None:
            r.update(has_prev=0, dt=0.0, disp=0.0, r_pos_cv=0.0, r_pos_ca=0.0,
                     r_spd=0.0, r_spd_pos=0.0, r_hed=0.0, implied_spd=0.0)
        else:
            dt = m.rcvTime - prev["rcvTime"]; dt = dt if dt > 0 else 1.0
            r_pos_cv = norm2(px - (prev["px"] + prev["sx"]*dt), py - (prev["py"] + prev["sy"]*dt))
            r_pos_ca = norm2(px - (prev["px"]+prev["sx"]*dt+0.5*prev["ax"]*dt*dt),
                             py - (prev["py"]+prev["sy"]*dt+0.5*prev["ay"]*dt*dt))
            r_spd = norm2(sx - (prev["sx"]+prev["ax"]*dt), sy - (prev["sy"]+prev["ay"]*dt))
            dispx, dispy = px - prev["px"], py - prev["py"]
            disp = norm2(dispx, dispy)
            r_spd_pos = norm2(sx - dispx/dt, sy - dispy/dt)
            implied_spd = norm2(dispx/dt, dispy/dt)
            if spd_mag > 0.5 and norm2(hx, hy) > 1e-6:
                cos = (sx*hx + sy*hy)/(spd_mag*norm2(hx, hy)); r_hed = 1.0 - max(-1.0, min(1.0, cos))
            else:
                r_hed = 0.0
            r.update(has_prev=1, dt=dt, disp=disp, r_pos_cv=r_pos_cv, r_pos_ca=r_pos_ca,
                     r_spd=r_spd, r_spd_pos=r_spd_pos, r_hed=r_hed, implied_spd=implied_spd)
        prev = dict(px=px, py=py, sx=sx, sy=sy, ax=ax, ay=ay, rcvTime=m.rcvTime)
        rows.append(r)
    return pd.DataFrame(rows)


def make_adaptive(seq, drift=(0.6, 0.6)):
    """Add constant drift velocity: claimed pos drifts, claimed speed shifted
    consistently so motion-model residuals stay benign. Returns crafted seq +
    per-message true-vs-claimed offset."""
    s = seq.sort_values("rcvTime").reset_index(drop=True).copy()
    t0 = s.rcvTime.iloc[0]
    trel = (s.rcvTime - t0).values
    s["px"] = s["px"].values + drift[0] * trel      # linear offset
    s["py"] = s["py"].values + drift[1] * trel
    s["sx"] = s["sx"].values + drift[0]             # consistent speed shift
    s["sy"] = s["sy"].values + drift[1]
    offset = norm2(drift[0] * trel, drift[1] * trel)
    return s, offset


def main():
    rf = joblib.load(os.path.join(MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(MODELS, "scalers.joblib"))
    feat_cols = sc["KINEMATIC"] + sc["RESIDUALS"]
    thr = sc["thr_rf"]

    # source benign trajectories from a scenario where the naive attack is easy
    scen = "ConstPos_1416"
    df = pd.read_parquet(os.path.join("data/features", scen + ".parquet"))
    df = df[df.attackerType >= 0]
    # restrict to held-out (test) senders to avoid scoring the detector's own
    # training trajectories.
    split_path = os.path.join("experiments/results", "split_map.json")
    if os.path.exists(split_path):
        sm = json.load(open(split_path))
        test_senders = {int(k) for k, v in sm.items() if v == "test"}
        df = df[df["sender"].isin(test_senders)]
    benign = df[df.is_attacker == 0]

    # naive attacker (real VeReMi) flag rate for reference
    Xn = sc["sc_all"].transform(df[df.is_attacker == 1][feat_cols].values)
    naive_flag = float((rf.predict_proba(Xn)[:, 1] >= thr).mean())

    # sample benign sequences long enough to accumulate offset
    grp = benign.sort_values("rcvTime").groupby(["receiver", "senderPseudo"])
    seqs = [g for _, g in grp if len(g) >= 20]
    rng = np.random.default_rng(42)
    if len(seqs) > 300:
        idx = rng.choice(len(seqs), 300, replace=False); seqs = [seqs[i] for i in idx]

    all_flag, all_offset, all_res = [], [], []
    for g in seqs:
        crafted, offset = make_adaptive(g[["px","py","sx","sy","ax","ay","hx","hy","rcvTime"]])
        feats = build_features(crafted)
        # only score messages where the attack has accumulated a meaningful offset
        mask = offset >= WIN_M
        if mask.sum() == 0:
            continue
        X = sc["sc_all"].transform(feats.loc[mask, feat_cols].values)
        p = rf.predict_proba(X)[:, 1]
        all_flag.extend((p >= thr).astype(int).tolist())
        all_offset.extend(offset[mask].tolist())
        all_res.extend(feats.loc[mask, "r_pos_cv"].tolist())

    adaptive_flag = float(np.mean(all_flag)) if all_flag else float("nan")
    out = {
        "note": "adaptive attacker rows are CONSTRUCTED (physics-respecting drift on "
                "real benign trajectories), not VeReMi ground truth.",
        "source_scenario": scen,
        "win_offset_m": WIN_M,
        "naive_attacker_flag_rate": round(naive_flag, 4),
        "adaptive_attacker_flag_rate": round(adaptive_flag, 4),
        "n_adaptive_messages_scored": len(all_flag),
        "adaptive_median_offset_m": round(float(np.median(all_offset)), 2) if all_offset else None,
        "adaptive_median_r_pos_cv": round(float(np.median(all_res)), 3) if all_res else None,
        "interpretation": "A physics-consistency detector cannot flag a spoof that "
                          "respects the motion model; the adaptive attacker sustains a "
                          "large position offset at a benign-level residual.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)); print("wrote", OUT)


if __name__ == "__main__":
    main()
