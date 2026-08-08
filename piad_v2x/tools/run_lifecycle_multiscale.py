"""Lifecycle outcomes with the MULTI-SCALE detector wired in, plus cluster-bootstrap CIs.

Closes the coherence gap the examiner gate found: the earlier lifecycle runs
(operating_point, time_to_isolate) were driven by the base kinematic detector
(rf.joblib), not the multi-scale detector of the detection chapter. This script trains a
multi-scale detector (KINEMATIC + multi-vehicle interaction invariants) STRICTLY on the
_1416 window, freezes it, and scores the held-out _0709 window (every _0709 sender unseen),
so the detector that drives the lifecycle is the same one the ablation chapter reports.
Training only on _1416 also removes a latent leakage concern in the base rf.joblib (which
was trained across all scenarios).

Both isolation-outcome metrics are computed at the pre-registered operating point
(TAU=0.15, K=3), each with a 95 percent cluster bootstrap over senders:
  - accepted-malicious-message reduction (arm D over arm C), persistent classes
  - benign false-revocation rate
  - time-to-isolate (Kaplan-Meier median, right-censoring never-isolated pseudonyms)

Run from the repo root:
    python piad_v2x/tools/run_lifecycle_multiscale.py --out experiments/results/lifecycle_multiscale.json
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from piad_v2x.models.relational import relational_features, KINEMATIC, REL_TRUE
from piad_v2x.lifecycle import closed_loop as cl
from piad_v2x.hypes_yaml import load_yaml

# Module defaults; overridden in main() from the --hypes_yaml configuration so
# that no hyperparameter is hardcoded on the execution path.
TAU, K = 0.15, 3
SEED = 42
N_BOOT = 1000
FEAT = KINEMATIC + REL_TRUE                       # multi-scale feature set
ORACLE = {"sender", "senderPseudo", "is_attacker", "attackerType", "messageID"}
PERSISTENT_0709 = ["ConstPos_0709", "RandomPos_0709", "DataReplay_0709",
                   "Disruptive_0709", "DelayedMessages_0709", "DoS_0709"]
TARGET_RECALL = 0.90                              # matches base-detector calibration
RF_ARGS = {"n_estimators": 120, "max_depth": 22, "max_train_rows": 1_500_000}
TRUST = {"alpha": cl.ALPHA, "t0": cl.T0, "n_min": cl.N_MIN}


def with_relational(path):
    df = pd.read_parquet(path)
    df = df[df.attackerType >= 0].reset_index(drop=True)
    rel = relational_features(df)
    return df.merge(rel, on="messageID", how="left")


def pick_threshold(y, p, target_recall=TARGET_RECALL):
    pos = float((y == 1).sum())
    if pos == 0:
        return 0.5
    order = np.argsort(-p); ys = y[order]; ps = p[order]
    rec = np.cumsum(ys) / pos
    idx = min(int(np.searchsorted(rec, target_recall)), len(ps) - 1)
    return float(ps[idx])


def train_multiscale_1416():
    """Train + calibrate the multi-scale detector on _1416 only (frozen)."""
    paths = sorted(glob.glob("data/features/*_1416.parquet"))
    if not paths:
        raise SystemExit("no _1416 scenarios in data/features")
    frames = [with_relational(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # sender split (train/val) on _1416; val only used to calibrate the threshold
    rng = np.random.default_rng(SEED)
    ug = np.array(df["sender"].astype(str).unique(), dtype=object); rng.shuffle(ug)
    val = set(ug[:int(0.2 * len(ug))])
    is_val = df["sender"].astype(str).isin(val).to_numpy()
    X = df[FEAT].fillna(0).to_numpy(); y = df["is_attacker"].to_numpy()
    Xtr, ytr = X[~is_val], y[~is_val]
    cap = RF_ARGS["max_train_rows"]
    if len(Xtr) > cap:
        idx = rng.choice(len(Xtr), size=cap, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    rf = RandomForestClassifier(n_estimators=RF_ARGS["n_estimators"],
                                max_depth=RF_ARGS["max_depth"], n_jobs=-1,
                                class_weight="balanced", random_state=SEED)
    rf.fit(Xtr, ytr)
    thr = pick_threshold(y[is_val], rf.predict_proba(X[is_val])[:, 1], TARGET_RECALL)
    print(f"trained multi-scale detector on _1416: {len(df):,} rows, "
          f"{len(ug)} senders, thr={thr:.4f}", flush=True)
    return rf, thr


def neutralisation(seqs, tau, K):
    out = {}
    for pseudo, (bstream, tstream, _isa, _snd) in seqs.items():
        T = TRUST["t0"]; below = 0; n = 0; t_neu = np.inf
        for b, t in zip(bstream, tstream):
            T = TRUST["alpha"] * T + (1 - TRUST["alpha"]) * b
            n += 1
            if n < TRUST["n_min"]:
                continue
            if T < tau:
                below += 1
                if below >= K:
                    t_neu = t; break
            else:
                below = 0
        out[pseudo] = t_neu
    return out


def km_median(times, censored):
    order = np.argsort(times)
    t = np.asarray(times, float)[order]; c = np.asarray(censored)[order]
    n = len(t); at_risk = n; S = 1.0; med = None; i = 0
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]:
            if not c[j]:
                d += 1
            j += 1
        if d > 0 and at_risk > 0:
            S *= (1 - d / at_risk)
        if med is None and S <= 0.5:
            med = float(t[i])
        at_risk -= (j - i); i = j
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypes_yaml", default="multiscale_rf.yaml",
                    help="experiment configuration under piad_v2x/hypes_yaml/")
    ap.add_argument("--out", default="experiments/results/lifecycle_multiscale.json")
    args = ap.parse_args()

    global TAU, K, SEED, N_BOOT, TARGET_RECALL, PERSISTENT_0709, RF_ARGS, TRUST
    cfg = load_yaml(args.hypes_yaml)
    SEED = cfg["data"]["seed"]
    TARGET_RECALL = cfg["detector"]["calibration"]["target_recall"]
    RF_ARGS = {"n_estimators": cfg["detector"]["args"]["n_estimators"],
               "max_depth": cfg["detector"]["args"]["max_depth"],
               "max_train_rows": cfg["detector"]["args"]["max_train_rows"]}
    tg = cfg["trust_gate"]
    TAU, K = tg["tau_revoke"], tg["k_consecutive"]
    TRUST = {"alpha": tg["alpha"], "t0": tg["t0"], "n_min": tg["n_min"]}
    N_BOOT = cfg["evaluation"]["bootstrap_resamples"]
    PERSISTENT_0709 = [s + cfg["data"]["eval_window"] for s in cfg["data"]["persistent_scenarios"]]
    print(f"loaded config '{cfg['name']}': detector={cfg['detector']['core_method']}, "
          f"TAU={TAU}, K={K}", flush=True)

    bad = [c for c in FEAT if c in ORACLE or c.startswith("true_")]
    if bad:
        raise SystemExit(f"oracle check failed: {bad}")
    rf, thr = train_multiscale_1416()

    # per attacker sender: accepted-malicious under arm C / arm D; per benign sender:
    # revoked flag; per attacker pseudonym: (t_neu, t0, n) for time-to-isolate.
    accC, accD = {}, {}            # keyed by (scenario, sender)
    benign_rev, benign_all = set(), set()
    tti, tti_cens = [], []         # finite time-to-isolate; censored flags
    tti_sender = []                # cluster id per tti entry (for bootstrap)
    flag_on_atk = []               # detector flag rate on attackers (context)

    for nm in PERSISTENT_0709:
        p = os.path.join("data/features", nm + ".parquet")
        if not os.path.exists(p):
            print("skip missing", nm); continue
        df = with_relational(p)
        df["p_mal"] = rf.predict_proba(df[FEAT].fillna(0).to_numpy())[:, 1]
        df["flagged"] = (df["p_mal"] >= thr).astype(int)
        seqs = {}
        for pseudo, g in df.sort_values("rcvTime").groupby("senderPseudo"):
            seqs[pseudo] = (1.0 - g["p_mal"].to_numpy(), g["rcvTime"].to_numpy(),
                            int(g["is_attacker"].iloc[0]), g["sender"].iloc[0])
        tneu = neutralisation(seqs, TAU, K)
        maxt = float(df["rcvTime"].max())

        atk = df[df.is_attacker == 1]
        flag_on_atk.append(float((atk.flagged == 1).mean()))
        atk = atk.assign(t_neu=atk.senderPseudo.map(tneu))
        atk["acc_off"] = (atk.flagged == 0).astype(int)
        atk["acc_on"] = ((atk.flagged == 0) & (atk.rcvTime < atk.t_neu.fillna(np.inf))).astype(int)
        for s, gg in atk.groupby("sender"):
            key = (nm, int(s))
            accC[key] = int(gg.acc_off.sum()); accD[key] = int(gg.acc_on.sum())

        benign = df[df.is_attacker == 0]
        for s, gg in benign.groupby("sender"):
            benign_all.add((nm, int(s)))
            if gg.senderPseudo.map(lambda p_: np.isfinite(tneu.get(p_, np.inf))).any():
                benign_rev.add((nm, int(s)))

        for pseudo, (b, tstream, isa, snd) in seqs.items():
            if isa != 1:
                continue
            tn = tneu.get(pseudo, np.inf); t0 = float(tstream[0])
            if np.isfinite(tn):
                tti.append(max(0.0, tn - t0)); tti_cens.append(0)
            else:
                tti.append(maxt - t0); tti_cens.append(1)
            tti_sender.append((nm, int(snd)))
        print(f"[{nm:22s}] scored ({len(seqs)} pseudonyms), flag_on_atk="
              f"{flag_on_atk[-1]:.3f}", flush=True)

    # ---- point estimates ----
    C = np.array([accC[k] for k in accC], float); D = np.array([accD[k] for k in accC], float)
    red_point = (C.sum() - D.sum()) / C.sum() if C.sum() > 0 else float("nan")
    fr_point = len(benign_rev) / max(1, len(benign_all))
    tti_med = km_median(tti, tti_cens)
    frac_iso = float(np.mean([1 - c for c in tti_cens])) if tti_cens else float("nan")

    # ---- cluster bootstrap over senders ----
    rng = np.random.default_rng(SEED)
    atk_keys = list(accC.keys()); ben_keys = list(benign_all)
    ttis = np.array(tti); ttic = np.array(tti_cens); ttik = tti_sender
    from collections import defaultdict
    by_sender = defaultdict(list)
    for i, k in enumerate(ttik):
        by_sender[k].append(i)
    tti_senders = list(by_sender.keys())

    red_bs, fr_bs, med_bs = [], [], []
    nA, nB, nT = len(atk_keys), len(ben_keys), len(tti_senders)
    for _ in range(N_BOOT):
        ia = rng.integers(0, nA, nA)
        cs = sum(accC[atk_keys[j]] for j in ia); ds = sum(accD[atk_keys[j]] for j in ia)
        red_bs.append((cs - ds) / cs if cs > 0 else 0.0)
        ib = rng.integers(0, nB, nB)
        rev = sum(1 for j in ib if ben_keys[j] in benign_rev)
        fr_bs.append(rev / max(1, nB))
        it = rng.integers(0, nT, nT)
        idxs = np.concatenate([by_sender[tti_senders[j]] for j in it])
        m = km_median(ttis[idxs], ttic[idxs])
        if m is not None:
            med_bs.append(m)

    def ci(a):
        a = np.array([x for x in a if x == x])
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if len(a) else None

    summary = {
        "protocol": "multi-scale detector (KINEMATIC + interaction invariants) trained on "
                    "_1416, frozen; scored on held-out _0709; TAU=0.15, K=3; "
                    "95% cluster bootstrap over senders (n=1000)",
        "detector_features": FEAT, "operating_point": {"TAU": TAU, "K": K},
        "detector_flag_rate_on_attackers_mean": round(float(np.mean(flag_on_atk)), 4),
        "accepted_malicious_reduction": {
            "point": round(red_point, 4), "ci95": ci(red_bs),
            "arm_C_accepted_malicious_total": int(C.sum()),
            "arm_D_accepted_malicious_total": int(D.sum()),
            "n_attacker_senders": nA},
        "false_revocation": {
            "point": round(fr_point, 4), "ci95": ci(fr_bs),
            "benign_revoked": len(benign_rev), "benign_total": len(benign_all)},
        "time_to_isolate": {
            "frac_isolated": round(frac_iso, 4),
            "km_median_s_point": round(tti_med, 3) if tti_med is not None else None,
            "km_median_s_ci95": ci(med_bs),
            "n_attacker_pseudonyms": len(tti)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("\n=== MULTI-SCALE LIFECYCLE (held-out _0709, TAU=0.15/K=3) ===")
    print("accepted-malicious reduction:", summary["accepted_malicious_reduction"])
    print("false-revocation:", summary["false_revocation"])
    print("time-to-isolate:", summary["time_to_isolate"])
    print("mean detector flag rate on attackers:", summary["detector_flag_rate_on_attackers_mean"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
