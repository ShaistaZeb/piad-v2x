"""Confirmatory test of the pre-registered lifecycle operating point on a held-out window.

Confirms the pre-registered operating point. The trade-curve point TAU_REV=0.15, K=3 was
found on the `_1416` window AFTER the benign-cost bound failed there, so it is exploratory on `_1416`. This
script tests it on the `_0709` window, which the point was NOT tuned on, using the frozen
detector trained on `_1416` senders (every `_0709` sender is therefore unseen - no split
filter is applied). Doubles as the cross-window generalisation check.

CONFIRMED iff, aggregated over the `_0709` persistent-attacker scenarios:
  mean harm reduction >= 0.20  AND  mean benign false-revocation <= 0.02.

Run from the repository root:
    python piad_v2x/tools/run_operating_point.py --out experiments/results/operating_point.json
"""
import argparse, json, os
import numpy as np
import pandas as pd
import joblib

from piad_v2x.lifecycle import closed_loop as cl

# _0709 persistent-attacker scenarios (mirror of the _1416 persistent set)
PERSISTENT_0709 = ["ConstPos_0709", "RandomPos_0709", "DataReplay_0709",
                   "Disruptive_0709", "DelayedMessages_0709", "DoS_0709"]
CONFIRM_TAU, CONFIRM_K = 0.15, 3     # pre-registered operating point
COMMITTED_TAU, COMMITTED_K = 0.35, 3  # the old committed point, for contrast
MDE, FR_BOUND = 0.20, 0.02
TAU_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
K_GRID = [3, 5, 8]


def prep_scenario(path, rf, sc):
    """Score once (all senders; detector never saw this window); return per-pseudonym
    sequences + attacker/benign bookkeeping for cheap trigger evaluation."""
    df = pd.read_parquet(path)
    df = df[df.attackerType >= 0].copy()
    if df.empty:
        return None
    df["p_mal"], thr = cl.score(df, rf, sc)
    df["flagged"] = (df["p_mal"] >= thr).astype(int)
    seqs = {}
    for pseudo, g in df.sort_values("rcvTime").groupby("senderPseudo"):
        seqs[pseudo] = (1.0 - g["p_mal"].to_numpy(), g["rcvTime"].to_numpy(),
                        int(g["is_attacker"].iloc[0]), g["sender"].iloc[0])
    atk = df[df.is_attacker == 1]
    return {
        "seqs": seqs,
        "atk_flagged": atk["flagged"].to_numpy(),
        "atk_rcv": atk["rcvTime"].to_numpy(),
        "atk_pseudo": atk["senderPseudo"].to_numpy(),
        "benign_senders": df.loc[df.is_attacker == 0, "sender"].unique(),
        "pseudo_sender_benign": {p: s for p, (_, _, isa, s) in seqs.items() if isa == 0},
    }


def neutralisation(seqs, tau, K):
    """t_neu per pseudonym under (tau, K), reusing closed_loop's ALPHA/N_MIN/T0."""
    out = {}
    for pseudo, (bstream, tstream, _isa, _snd) in seqs.items():
        T = cl.T0; below = 0; n = 0; t_neu = np.inf
        for b, t in zip(bstream, tstream):
            T = cl.ALPHA * T + (1 - cl.ALPHA) * b
            n += 1
            if n < cl.N_MIN:
                continue
            if T < tau:
                below += 1
                if below >= K:
                    t_neu = t; break
            else:
                below = 0
        out[pseudo] = t_neu
    return out


def metrics_for_setting(prep, tau, K):
    tneu = neutralisation(prep["seqs"], tau, K)
    fl = prep["atk_flagged"]; rcv = prep["atk_rcv"]
    tn = np.array([tneu.get(p, np.inf) for p in prep["atk_pseudo"]])
    accC = int((fl == 0).sum())
    accD = int(((fl == 0) & (rcv < tn)).sum())
    reduction = (accC - accD) / accC if accC > 0 else float("nan")
    benign_ps = prep["pseudo_sender_benign"]
    revoked = {benign_ps[p] for p, t in tneu.items()
               if p in benign_ps and np.isfinite(t)}
    n_benign = max(1, len(prep["benign_senders"]))
    return reduction, len(revoked) / n_benign, len(revoked), n_benign


def aggregate(preps, tau, K):
    reds, frs, rev, nb = [], [], 0, 0
    for pr in preps.values():
        r, fr, nrev, nben = metrics_for_setting(pr, tau, K)
        if r == r:
            reds.append(r)
        frs.append(fr); rev += nrev; nb += nben
    return {
        "tau": tau, "K": K,
        "mean_persistent_reduction": round(float(np.mean(reds)), 4) if reds else None,
        "mean_false_revocation": round(float(np.mean(frs)), 4) if frs else None,
        "benign_revoked_total": int(rev), "benign_senders_total": int(nb),
        "pooled_false_revocation": round(rev / max(1, nb), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=PERSISTENT_0709)
    ap.add_argument("--out", default="experiments/results/operating_point.json")
    args = ap.parse_args()
    rf = joblib.load(os.path.join(cl.MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(cl.MODELS, "scalers.joblib"))

    preps = {}
    for nm in args.scenarios:
        p = os.path.join("data/features", nm + ".parquet")
        if not os.path.exists(p):
            print("skip missing", nm); continue
        pr = prep_scenario(p, rf, sc)
        if pr is not None:
            preps[nm] = pr
            print(f"scored {nm} ({len(pr['seqs'])} pseudonyms, "
                  f"{len(pr['benign_senders'])} benign senders)", flush=True)
    if not preps:
        raise SystemExit("no _0709 scenarios found; extract them first (see data/README.md)")

    confirm = aggregate(preps, CONFIRM_TAU, CONFIRM_K)
    committed = aggregate(preps, COMMITTED_TAU, COMMITTED_K)
    grid = [aggregate(preps, tau, K) for K in K_GRID for tau in TAU_GRID]

    passed = (confirm["mean_persistent_reduction"] is not None
              and confirm["mean_persistent_reduction"] >= MDE
              and confirm["mean_false_revocation"] is not None
              and confirm["mean_false_revocation"] <= FR_BOUND)
    summary = {
        "protocol": "operating-point confirmation",
        "window": "_0709 (held-out; detector trained on _1416, all senders unseen)",
        "mde_reduction": MDE, "fr_bound": FR_BOUND,
        "pre_registered_point": confirm,
        "committed_point_for_contrast": committed,
        "grid": grid,
        "confirmed": bool(passed),
        "verdict": ("CONFIRMED: tau=0.15/K=3 transfers to the held-out _0709 window "
                    "(reduction>=0.20 and false-revocation<=0.02)"
                    if passed else
                    "NOT CONFIRMED: the point does not satisfy both bounds on _0709; "
                    "report as a _1416-specific tuning"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\npre-registered point tau=0.15/K=3 on _0709: {confirm}")
    print("verdict:", summary["verdict"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
