#!/usr/bin/env python3
"""External validity of the COMMITTED coupling on VeReMi NextGen (InTAS).

External-validity evaluation on the VeReMi NextGen (InTAS) benchmark.
Retrains the SAME pipeline on NextGen's own Train split: pooled multi-scale detector
(leakage-safe sender split, recall-0.90 threshold) AND the trust gate TAU, both re-calibrated to
NextGen. TAU is tuned on held-out Train benign to a matched benign false-revocation target (0.5%),
tuned to the FR target and NOT to maximize the reduction. Evaluated on each attack's Test split at
K=3. Reuses the committed neutralisation/threshold/KM-median code verbatim.

Primary metric: reduction-versus-no-defense = 1 - accepted-malicious rate under the full coupled
system (stable where detection is near-perfect). reduction-versus-detection-only (the committed
Extension metric) is reported where defined.
"""
import argparse, importlib.util, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from piad_v2x.models.relational import KINEMATIC, REL_TRUE

FEAT = KINEMATIC + REL_TRUE
K, SEED, TARGET_RECALL, TARGET_FR = 3, 42, 0.90, 0.005
FEATDIR = "data/features_nextgen"

_s = importlib.util.spec_from_file_location("rlm", os.path.join("piad_v2x", "tools", "run_lifecycle_multiscale.py"))
LMS = importlib.util.module_from_spec(_s); _s.loader.exec_module(LMS)
neutralisation, pick_threshold, km_median = LMS.neutralisation, LMS.pick_threshold, LMS.km_median

PERSISTENT = ["constantPositionOffset", "randomPositionOffset", "zeroSpeedReport",
              "dataReplay", "timeDelayAttack", "dosAttack"]
DETECTABLE = ["randomPositionOffset", "zeroSpeedReport", "dataReplay", "timeDelayAttack", "dosAttack"]


def seqs_of(df):
    s = {}
    for pseudo, g in df.sort_values("rcvTime").groupby("senderPseudo"):
        s[pseudo] = (1.0 - g["p_mal"].to_numpy(), g["rcvTime"].to_numpy(),
                     int(g["is_attacker"].iloc[0]), g["sender"].iloc[0])
    return s


def train_detector(scenario):
    frames = [pd.read_parquet(f"{FEATDIR}/InTAS_{scenario}_{a}_Train.parquet")
              for a in PERSISTENT if os.path.exists(f"{FEATDIR}/InTAS_{scenario}_{a}_Train.parquet")]
    df = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(SEED)
    ug = np.array(sorted(df["sender"].unique()), dtype=object); rng.shuffle(ug)
    val = set(ug[:int(0.2 * len(ug))])
    isval = df["sender"].isin(val).to_numpy()
    X = df[FEAT].fillna(0).to_numpy(); y = df["is_attacker"].to_numpy()
    Xtr, ytr = X[~isval], y[~isval]
    cap = 1_500_000
    if len(Xtr) > cap:
        idx = rng.choice(len(Xtr), size=cap, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    rf = RandomForestClassifier(n_estimators=120, max_depth=22, n_jobs=-1,
                                class_weight="balanced", random_state=SEED).fit(Xtr, ytr)
    thr = pick_threshold(y[isval], rf.predict_proba(X[isval])[:, 1])
    print(f"trained multi-scale on {scenario} (pooled persistent Train), thr={thr:.4f}", flush=True)
    return rf, thr


def calibrate_tau(rf, scenario, attack, target_fr=TARGET_FR):
    """Per-attack TAU on that attack's own Train benign, to the matched false-revocation target.
    Highest (most sensitive) TAU whose benign FR stays <= target; if none, the strictest TAU and
    its (over-target) FR is reported honestly. Tuned to FR, NOT to the reduction."""
    df = pd.read_parquet(f"{FEATDIR}/InTAS_{scenario}_{attack}_Train.parquet")
    df["p_mal"] = rf.predict_proba(df[FEAT].fillna(0).to_numpy())[:, 1]
    ben = df[df.is_attacker == 0]
    seqs = seqs_of(ben)
    ntot = ben["sender"].nunique()
    frs = []
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        tneu = neutralisation(seqs, tau, K)
        rev = {seqs[p][3] for p, tn in tneu.items() if np.isfinite(tn)}
        frs.append((tau, len(rev) / max(1, ntot)))
    ok = [t for t, f in frs if f <= target_fr]
    tau = max(ok) if ok else frs[0][0]
    fr = dict(frs)[tau]
    return tau, round(fr, 4)


def evaluate(rf, thr, tau, path):
    df = pd.read_parquet(path)
    df["p_mal"] = rf.predict_proba(df[FEAT].fillna(0).to_numpy())[:, 1]
    df["flagged"] = (df["p_mal"] >= thr).astype(int)
    seqs = seqs_of(df)
    tneu = neutralisation(seqs, tau, K)
    maxt = float(df["rcvTime"].max())
    atk = df[df.is_attacker == 1].assign(t_neu=lambda d: d.senderPseudo.map(tneu))
    atk["acc_on"] = ((atk.flagged == 0) & (atk.rcvTime < atk.t_neu.fillna(np.inf))).astype(int)
    atk["acc_off"] = (atk.flagged == 0).astype(int)
    tot, on, off = [], [], []
    for s, gg in atk.groupby("sender"):
        tot.append(len(gg)); on.append(int(gg.acc_on.sum())); off.append(int(gg.acc_off.sum()))
    tot, on, off = np.array(tot, float), np.array(on, float), np.array(off, float)
    red_nodef = 1 - on.sum() / tot.sum()
    red_det = (off.sum() - on.sum()) / off.sum() if off.sum() > 0 else None
    rng = np.random.default_rng(SEED); n = len(tot); b = []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        b.append(1 - on[i].sum() / tot[i].sum() if tot[i].sum() > 0 else np.nan)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    ben = df[df.is_attacker == 0]
    brev = sum(1 for _, gg in ben.groupby("sender")
               if gg.senderPseudo.map(lambda p: np.isfinite(tneu.get(p, np.inf))).any())
    fr = brev / max(1, ben["sender"].nunique())
    tti, cens = [], []
    for pseudo, (bs, ts, isa, snd) in seqs.items():
        if isa != 1: continue
        tn = tneu.get(pseudo, np.inf); t0 = float(ts[0])
        if np.isfinite(tn): tti.append(max(0.0, tn - t0)); cens.append(0)
        else: tti.append(maxt - t0); cens.append(1)
    fiso = float(np.mean([1 - c for c in cens])) if cens else float("nan")
    kmm = km_median(np.array(tti), np.array(cens)) if tti else None
    return dict(acc_malicious_rate=round(float(on.sum() / tot.sum()), 4),
                reduction_vs_nodefense=round(float(red_nodef), 4), ci95=[round(lo, 4), round(hi, 4)],
                reduction_vs_detection=(round(float(red_det), 4) if red_det is not None else None),
                false_revocation=round(fr, 4), frac_isolated=round(fiso, 4),
                km_median_s=(round(float(kmm), 2) if kmm is not None else None),
                flag_rate=round(float((atk.flagged == 1).mean()), 4), n_attacker_senders=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="highway_2")
    ap.add_argument("--attacks", nargs="*", default=PERSISTENT + ["trafficCongestionSybil"])
    ap.add_argument("--out", default="experiments/results/nextgen_validation.json")
    args = ap.parse_args()
    rf, thr = train_detector(args.scenario)
    rows = {}
    for a in args.attacks:
        p = f"{FEATDIR}/InTAS_{args.scenario}_{a}_Test.parquet"
        if not os.path.exists(p): print("skip missing", a); continue
        tau, cal_fr = calibrate_tau(rf, args.scenario, a)
        r = evaluate(rf, thr, tau, p); r["tau"] = tau; r["cal_fr"] = cal_fr
        rows[a] = r
        tag = "DETECTABLE" if a in DETECTABLE else ("SYBIL-bnd" if "Sybil" in a else "hard-ConstPos")
        matched = "" if cal_fr <= TARGET_FR else " [FR-TARGET UNMET]"
        print(f"[{a:24s} {tag:13s}] stop={r['reduction_vs_nodefense']:.3f} CI{r['ci95']} "
              f"tau={tau} testFR={r['false_revocation']:.4f} iso={r['frac_isolated']:.3f}{matched}",
              flush=True)
    det = [rows[a]["reduction_vs_nodefense"] for a in DETECTABLE if a in rows]
    print(f"\nDETECTABLE-set reduction-vs-no-defense: min={min(det):.3f} median={np.median(det):.3f} "
          f"(floor 0.50)")
    # TAU is calibrated per attack (see by_attack[*].tau); there is no single global TAU.
    json.dump({"scenario": args.scenario,
               "operating_point": {"TAU": "per-attack (see by_attack[*].tau)", "K": K, "thr": thr,
                                   "target_fr": TARGET_FR}, "by_attack": rows},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
