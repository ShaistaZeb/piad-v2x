#!/usr/bin/env python3
"""
demo_trace.py - per-attacker trust trace for live demos.

Replays one attacker pseudonym message by message and shows its physics-grounded
trust decaying under the EWMA aggregator, crossing the revocation threshold tau,
and the moment the trust-gated lifecycle neutralises (revokes) it. Then prints
how many malicious messages the loop accepted before isolation, versus what
per-message detection alone would have let through.

Uses the same trained detector and the same frozen coupling constants as
piad_v2x/lifecycle/closed_loop.py, so the trace matches the committed pipeline.

Usage (from the repository root, with the venv active):
    python piad_v2x/tools/demo_trace.py                       # auto-pick a clean example
    python piad_v2x/tools/demo_trace.py --scenario DataReplay_1416
    python piad_v2x/tools/demo_trace.py --scenario DoS_1416 --pseudo 123456
    python piad_v2x/tools/demo_trace.py --scenario DataReplay_1416 --list
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

MODELS = "experiments/results/models"
# frozen coupling constants (identical to closed_loop.py)
ALPHA = 0.8
N_MIN = 5
T0 = 0.5
TAU_REV = 0.15
K = 3


def load_models():
    rf = joblib.load(os.path.join(MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(MODELS, "scalers.joblib"))
    return rf, sc


def score(df, rf, sc):
    X = sc["sc_all"].transform(df[sc["KINEMATIC"] + sc["RESIDUALS"]].values)
    return rf.predict_proba(X)[:, 1], sc["thr_rf"]


def load_scenario(name, rf, sc):
    path = os.path.join("data/features", name + ".parquet")
    if not os.path.exists(path):
        raise SystemExit(f"scenario not found: {path}")
    df = pd.read_parquet(path)
    df = df[df.attackerType >= 0].copy()
    # match closed_loop: evaluate on held-out (test) senders only
    split_path = os.path.join("experiments/results", "split_map.json")
    if os.path.exists(split_path):
        sm = json.load(open(split_path))
        test = {int(k) for k, v in sm.items() if v == "test"}
        df = df[df["sender"].isin(test)].copy()
    df["p_mal"], thr = score(df, rf, sc)
    df["flagged"] = (df["p_mal"] >= thr).astype(int)
    return df, thr


def trace_pseudo(g):
    """Replay one pseudonym's message stream. Returns per-step records and the
    neutralisation time (inf if never neutralised). Mirrors closed_loop's
    neutralisation_times exactly: EWMA trust, cold start of N_MIN, then revoke
    after K consecutive updates below tau."""
    g = g.sort_values("rcvTime")
    T = T0
    below = 0
    n = 0
    t_neu = np.inf
    steps = []
    for p_mal, t, flagged in zip(g["p_mal"].values, g["rcvTime"].values, g["flagged"].values):
        b = 1.0 - p_mal
        T = ALPHA * T + (1 - ALPHA) * b
        n += 1
        if n >= N_MIN and T < TAU_REV:
            below += 1
        else:
            below = 0
        if below >= K and not np.isfinite(t_neu):
            t_neu = t
        steps.append((n, float(t), float(p_mal), float(T), int(flagged), below, np.isfinite(t_neu)))
        if np.isfinite(t_neu):
            break  # stop at revocation for a clean demo
    return steps, t_neu


def bar(T, width=24):
    fill = int(round(max(0.0, min(1.0, T)) * width))
    return "#" * fill + "-" * (width - fill)


def pick_pseudo(df):
    """Auto-pick a neutralised attacker pseudonym that makes a clean trace
    (8 to 40 updates); fall back to any neutralised one, then any at all."""
    atk = df[df.is_attacker == 1]
    fallback = None
    for pseudo, g in atk.groupby("senderPseudo"):
        _, t_neu = trace_pseudo(g)
        if np.isfinite(t_neu) and 8 <= len(g) <= 40:
            return pseudo
        if np.isfinite(t_neu) and fallback is None:
            fallback = pseudo
    return fallback if fallback is not None else atk["senderPseudo"].iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="DataReplay_1416",
                    help="scenario name under data/features/ (default: DataReplay_1416)")
    ap.add_argument("--pseudo", type=int, default=None,
                    help="attacker pseudonym id (default: auto-pick a clean example)")
    ap.add_argument("--list", action="store_true",
                    help="list candidate attacker pseudonyms and exit")
    args = ap.parse_args()

    rf, sc = load_models()
    df, thr = load_scenario(args.scenario, rf, sc)
    atk = df[df.is_attacker == 1]
    if atk.empty:
        raise SystemExit(f"no attacker messages in {args.scenario}")

    if args.list:
        print(f"attacker pseudonyms in {args.scenario} (rows, neutralised?):")
        for pseudo, g in list(atk.groupby("senderPseudo"))[:40]:
            _, t_neu = trace_pseudo(g)
            print(f"  {pseudo:>12}  rows={len(g):<4} neutralised={'yes' if np.isfinite(t_neu) else 'no'}")
        return

    pseudo = args.pseudo if args.pseudo is not None else pick_pseudo(df)
    g = atk[atk.senderPseudo == pseudo]
    if g.empty:
        raise SystemExit(f"pseudonym {pseudo} not found among attackers in {args.scenario}")

    steps, t_neu = trace_pseudo(g)
    t_first = float(g["rcvTime"].min())

    print()
    print(f"  PIAD-V2X live trace   scenario {args.scenario}   attacker pseudonym {pseudo}")
    print(f"  detector flags a message when p_mal >= thr={thr:.3f}")
    print(f"  revoke the pseudonym when trust < tau={TAU_REV} for K={K} updates (after N_min={N_MIN})")
    print(f"  trust starts at T0={T0}, EWMA alpha={ALPHA} over the physics-grounded per-message score")
    print("  " + "-" * 82)
    print(f"  {'#':>3} {'t(s)':>8} {'p_mal':>6} {'trust':>6}  {'trust bar':<26} {'flag':>4}  status")
    print("  " + "-" * 82)
    for (n, t, p_mal, T, flagged, below, neutralised) in steps:
        status = ""
        if n >= N_MIN and T < TAU_REV:
            status = f"below tau ({below}/{K})"
        if neutralised:
            status = ">>> NEUTRALISED / PSEUDONYM REVOKED <<<"
        flag_s = "MAL" if flagged else "."
        print(f"  {n:>3} {t:>8.1f} {p_mal:>6.2f} {T:>6.2f}  [{bar(T)}] {flag_s:>4}  {status}")
    print("  " + "-" * 82)

    acc_C = int((g.flagged == 0).sum())                                 # detect-only
    acc_D = int(((g.flagged == 0) & (g.rcvTime < t_neu)).sum())         # full coupling
    if np.isfinite(t_neu):
        print(f"  REVOKED at t={t_neu:.1f}s   time-to-isolate = {t_neu - t_first:.1f}s from its first message")
    else:
        print("  NOT neutralised within its lifetime (short-lived Sybil / benign-until-stop boundary)")
    print("  malicious messages this attacker got ACCEPTED into the safety picture:")
    print(f"     per-message detection only (arm C): {acc_C}")
    print(f"     full trust-gated coupling  (arm D): {acc_D}   <- revoking the identity stops the rest")
    print()


if __name__ == "__main__":
    main()
