#!/usr/bin/env python3
"""
Closed-loop trust-gated pseudonym-lifecycle simulation.

Uses the frozen coupling specification and the trained detector
(RF) to score every message in a scenario, aggregates per-pseudonym trust over
time, and compares four arms on:
  - accepted malicious messages before neutralisation (primary, off causal path)
  - attacker effective uptime and fraction neutralised (secondary)
  - honest-vehicle pseudonym lifetime (privacy)

The load-bearing comparison is D (full coupling) vs C (coupling-OFF): both flag
per message; only D routes aggregated trust into revocation + adaptive rotation.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, joblib

MODELS = "experiments/results/models"
OUT = "experiments/results/closed_loop.json"
# coupling constants (frozen a priori)
ALPHA = 0.8; N_MIN = 5; T0 = 0.5; TAU_REV = 0.15; K = 3; C_BASE = 60.0; BETA = 1.0  # TAU_REV=0.15 = pre-registered operating point (was 0.35; aligned 2026-07-18)


def score(df, rf, sc):
    X = sc["sc_all"].transform(df[sc["KINEMATIC"] + sc["RESIDUALS"]].values)
    p = rf.predict_proba(X)[:, 1]
    return p, sc["thr_rf"]


def neutralisation_times(df):
    """Per senderPseudo: time it is neutralised (T<TAU for K consecutive updates
    after cold start), else inf."""
    out = {}
    for pseudo, g in df.sort_values("rcvTime").groupby("senderPseudo"):
        T = T0; below = 0; n = 0; t_neu = np.inf
        for b, t in zip(1.0 - g["p_mal"].values, g["rcvTime"].values):
            T = ALPHA * T + (1 - ALPHA) * b
            n += 1
            if n < N_MIN:
                continue
            if T < TAU_REV:
                below += 1
                if below >= K:
                    t_neu = t; break
            else:
                below = 0
        out[pseudo] = t_neu
    return out


def run_scenario(scen_path, rf, sc, test_senders):
    df = pd.read_parquet(scen_path)
    df = df[df.attackerType >= 0].copy()
    # restrict to HELD-OUT senders: the detector was trained on other senders'
    # rows, so scoring only test senders avoids train-on-test contamination.
    if test_senders is not None:
        df = df[df["sender"].isin(test_senders)].copy()
    df["p_mal"], thr = score(df, rf, sc)
    df["flagged"] = (df["p_mal"] >= thr).astype(int)
    tneu = neutralisation_times(df)
    df["t_neu"] = df["senderPseudo"].map(tneu)

    atk = df[df.is_attacker == 1]
    n_atk = len(atk)

    # Variant A fixed: no reject, no revocation -> all malicious accepted
    accA = n_atk
    # Variant C coupling-OFF: reject per-message flagged -> accepted = unflagged attacker msgs
    accC = int(((atk.flagged == 0)).sum())
    # Variant D full coupling: reject flagged OR from neutralised pseudonym (rcvTime>=t_neu)
    accD = int(((atk.flagged == 0) & (atk.rcvTime < atk.t_neu)).sum())

    # secondary: attacker pseudonyms ever neutralised
    atk_pseudos = atk["senderPseudo"].unique()
    neu_pseudos = [p for p in atk_pseudos if np.isfinite(tneu.get(p, np.inf))]
    frac_neu = len(neu_pseudos) / max(1, len(atk_pseudos))
    # effective uptime: first malicious msg time -> neutralisation, per pseudonym
    uptimes = []
    for p, g in atk.groupby("senderPseudo"):
        t0 = g["rcvTime"].min(); tn = tneu.get(p, np.inf)
        uptimes.append((min(tn, g["rcvTime"].max()) - t0) if np.isfinite(tn)
                       else (g["rcvTime"].max() - t0))
    mean_uptime = float(np.mean(uptimes)) if uptimes else 0.0

    # privacy: honest-vehicle pseudonym lifetime = the target rotation cadence the
    # coupling emits. hostile_frac is TRUST-DRIVEN (fraction of a receiver's
    # observed pseudonyms whose locally-aggregated benign-prob < TAU_REV), NOT the
    # oracle is_attacker. Restricted to honest receivers (vehId not in attacker set).
    attacker_vehids = set(atk["sender"].unique())
    hostile_fracs, dens = [], []
    for r, g in df.groupby("receiver"):
        if r in attacker_vehids:      # only honest vehicles' privacy matters
            continue
        # per observed pseudonym: mean benign prob this receiver saw
        pt = g.groupby("senderPseudo")["p_mal"].mean()
        if len(pt) == 0:
            continue
        hostile = (1.0 - pt) < TAU_REV     # low trust = hostile-looking peer
        hostile_fracs.append(float(hostile.mean()))
        dens.append(len(pt))
    hostile_fracs = np.array(hostile_fracs); dens = np.array(dens, dtype=float)
    dens_norm = dens / dens.max() if len(dens) and dens.max() > 0 else dens
    life_A = C_BASE
    life_B = float(np.mean(C_BASE / (1 + BETA * dens_norm))) if len(dens_norm) else C_BASE
    life_D = float(np.mean(C_BASE / (1 + BETA * hostile_fracs))) if len(hostile_fracs) else C_BASE

    return {
        "scenario": os.path.basename(scen_path).replace(".parquet", ""),
        "n_attacker_msgs": int(n_atk),
        "accepted_malicious": {"A_fixed": int(accA), "C_coupling_off": int(accC),
                               "D_full_coupling": int(accD)},
        "accepted_rate": {"A_fixed": round(accA / max(1, n_atk), 4),
                          "C_coupling_off": round(accC / max(1, n_atk), 4),
                          "D_full_coupling": round(accD / max(1, n_atk), 4)},
        "D_vs_C_relative_reduction": round((accC - accD) / max(1, accC), 4),
        "frac_attacker_pseudos_neutralised": round(frac_neu, 4),
        "mean_attacker_effective_uptime_s": round(mean_uptime, 2),
        "privacy_mean_pseudo_lifetime_s": {"A_fixed": round(life_A, 2),
                                           "B_context_adaptive": round(life_B, 2),
                                           "D_full_coupling": round(life_D, 2)},
        "privacy_D_no_worse_than_A": bool(life_D <= life_A * 1.05),
        "detector_flag_rate_on_attackers": round(float((atk.flagged == 1).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="scenario names (default: a representative subset)")
    args = ap.parse_args()
    rf = joblib.load(os.path.join(MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(MODELS, "scalers.joblib"))
    split_path = os.path.join("experiments/results", "split_map.json")
    test_senders = None
    if os.path.exists(split_path):
        sm = json.load(open(split_path))
        test_senders = {int(k) for k, v in sm.items() if v == "test"}
        print(f"restricting to {len(test_senders)} held-out (test) senders", flush=True)
    default = ["ConstPos_1416", "RandomPos_1416", "DataReplay_1416",
               "DataReplaySybil_1416", "Disruptive_1416", "EventualStop_1416",
               "DelayedMessages_1416", "DoS_1416"]
    names = args.scenarios or default
    results = []
    for nm in names:
        p = os.path.join("data/features", nm + ".parquet")
        if not os.path.exists(p):
            print("skip missing", p); continue
        r = run_scenario(p, rf, sc, test_senders)
        results.append(r)
        print(f"{r['scenario']:22s} acc% A/C/D = "
              f"{r['accepted_rate']['A_fixed']:.3f}/{r['accepted_rate']['C_coupling_off']:.3f}/"
              f"{r['accepted_rate']['D_full_coupling']:.3f}  DvsC_red={r['D_vs_C_relative_reduction']:.3f}  "
              f"neu%={r['frac_attacker_pseudos_neutralised']:.2f}  "
              f"flag%={r['detector_flag_rate_on_attackers']:.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
