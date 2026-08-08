"""Closed-loop isolation outcomes of the trust-gated lifecycle.

Runs the confirmatory closed-loop isolation-outcome protocol. Run from
the repository root (after `pip install -e .`):

    python piad_v2x/tools/run_isolation_outcomes.py --out experiments/results/isolation_outcomes.json

The coupling-OFF (variant C: per-message rejection only) and coupling-ON (variant D: rejection
plus trust-driven neutralisation) accepted-malicious counts come from the tested
`piad_v2x.lifecycle.closed_loop` logic. This runner adds the three things the
pre-registration requires and the exploratory closed_loop.json lacked:
  (1) per-SENDER accepted-message counts and a cluster bootstrap over senders (CI),
  (2) the false-revocation rate on BENIGN senders (the benign-cost side),
  (3) the persistent-vs-Sybil split and the 20% MDE decision rule + Holm correction.

Primary metric (off the causal path, per the pre-reg): accepted malicious messages, i.e.
attacker BSMs that pass the trust gate before that pseudonym is neutralised. The
coupling-ON vs OFF relative reduction is (accC - accD) / accC.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import joblib

from piad_v2x.lifecycle import closed_loop as cl

PERSISTENT = {"ConstPos", "RandomPos", "DataReplay", "Disruptive", "DelayedMessages", "DoS"}
SYBIL = {"DataReplaySybil", "GridSybil", "DoSRandomSybil", "DoSDisruptiveSybil"}
MDE = 0.20                # pre-registered minimum relative reduction
FR_BOUND = 0.02           # pre-registered false-revocation bound
N_BOOT = 1000
SEED = 42


def per_sender_counts(scen_path, rf, sc, test_senders):
    """Per attacker SENDER: accepted-malicious under coupling-OFF (variant C) and
    coupling-ON (variant D), plus the benign false-revocation tally. Replicates the
    acceptance logic in closed_loop.run_scenario, grouped by sender for bootstrapping."""
    df = pd.read_parquet(scen_path)
    df = df[df.attackerType >= 0].copy()
    if test_senders is not None:
        df = df[df["sender"].isin(test_senders)].copy()
    if df.empty:
        return None
    df["p_mal"], thr = cl.score(df, rf, sc)
    df["flagged"] = (df["p_mal"] >= thr).astype(int)
    tneu = cl.neutralisation_times(df)
    df["t_neu"] = df["senderPseudo"].map(tneu)

    atk = df[df.is_attacker == 1].copy()
    # accepted = malicious message NOT rejected. OFF: reject flagged. ON: reject flagged
    # OR from a neutralised pseudonym (rcvTime >= t_neu).
    atk["acc_off"] = (atk.flagged == 0).astype(int)
    atk["acc_on"] = ((atk.flagged == 0) & (atk.rcvTime < atk.t_neu.fillna(np.inf))).astype(int)
    per = atk.groupby("sender")[["acc_off", "acc_on"]].sum()
    accC = per["acc_off"].to_dict()
    accD = per["acc_on"].to_dict()

    # false revocation: benign SENDERS with any pseudonym neutralised
    benign = df[df.is_attacker == 0]
    b_send = benign["sender"].unique()
    wrongly = 0
    for s, g in benign.groupby("sender"):
        if g["senderPseudo"].map(lambda p: np.isfinite(tneu.get(p, np.inf))).any():
            wrongly += 1
    return {"accC": accC, "accD": accD,
            "n_benign_senders": int(len(b_send)), "n_benign_revoked": int(wrongly)}


def bootstrap_reduction(accC, accD, rng):
    senders = list(accC.keys())
    if not senders:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    C = np.array([accC[s] for s in senders], float)
    D = np.array([accD[s] for s in senders], float)
    point = (C.sum() - D.sum()) / C.sum() if C.sum() > 0 else float("nan")
    reds = []
    n = len(senders)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        cs, ds = C[idx].sum(), D[idx].sum()
        reds.append((cs - ds) / cs if cs > 0 else 0.0)
    reds = np.array(reds)
    ci = (float(np.percentile(reds, 2.5)), float(np.percentile(reds, 97.5)))
    p_one_sided = float((reds <= 0).mean())   # H0: reduction <= 0
    return point, ci, p_one_sided


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - s), c + s)


def holm(pairs):
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs); adj = {}; running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pairs[i][1])
        running = max(running, val)
        adj[pairs[i][0]] = round(running, 4)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/results/isolation_outcomes.json")
    args = ap.parse_args()

    rf = joblib.load(os.path.join(cl.MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(cl.MODELS, "scalers.joblib"))
    sm_path = "experiments/results/split_map.json"
    test_senders = None
    if os.path.exists(sm_path):
        sm = json.load(open(sm_path))
        test_senders = {int(k) for k, v in sm.items() if v == "test"}

    default = ["ConstPos_1416", "RandomPos_1416", "DataReplay_1416", "Disruptive_1416",
               "DelayedMessages_1416", "DoS_1416", "DataReplaySybil_1416", "EventualStop_1416"]
    names = args.scenarios or default
    rng = np.random.default_rng(SEED)
    rows = {}
    pvals = []
    for nm in names:
        p = os.path.join("data/features", nm + ".parquet")
        if not os.path.exists(p):
            print("skip missing", nm); continue
        base = nm.split("_")[0]
        regime = "persistent" if base in PERSISTENT else ("sybil" if base in SYBIL else "other")
        c = per_sender_counts(p, rf, sc, test_senders)
        if c is None:
            print("skip empty", nm); continue
        point, ci, pval = bootstrap_reduction(c["accC"], c["accD"], rng)
        fr = c["n_benign_revoked"] / max(1, c["n_benign_senders"])
        fr_ci = wilson(c["n_benign_revoked"], c["n_benign_senders"])
        rows[nm] = {
            "regime": regime,
            "reduction_point": round(point, 4) if point == point else None,
            "reduction_ci95": [round(ci[0], 4), round(ci[1], 4)] if ci[0] == ci[0] else None,
            "p_one_sided": round(pval, 4),
            "false_revocation_rate": round(fr, 4),
            "false_revocation_ci95": [round(fr_ci[0], 4), round(fr_ci[1], 4)],
            "n_attacker_senders": len(c["accC"]),
            "n_benign_senders": c["n_benign_senders"],
        }
        if regime == "persistent":
            pvals.append((nm, pval))
        print(f"[{nm:22s} {regime:10s}] reduction={rows[nm]['reduction_point']} "
              f"CI={rows[nm]['reduction_ci95']} p={pval:.3f} "
              f"false_revoke={fr:.4f} CI={rows[nm]['false_revocation_ci95']}", flush=True)

    holm_adj = holm(pvals) if pvals else {}
    for nm, padj in holm_adj.items():
        rows[nm]["p_holm"] = padj
        r = rows[nm]
        supported = (r["reduction_ci95"] and r["reduction_ci95"][0] > 0
                     and r["reduction_point"] is not None and r["reduction_point"] >= MDE
                     and padj < 0.05)
        rows[nm]["reduction_supported"] = bool(supported)
        rows[nm]["false_revocation_within_bound"] = bool(r["false_revocation_ci95"][1] <= FR_BOUND)

    summary = {
        "protocol": "closed-loop isolation outcomes",
        "mde_relative_reduction": MDE, "false_revoke_bound": FR_BOUND, "n_bootstrap": N_BOOT,
        "primary_metric": "accepted malicious messages; reduction = (accC - accD)/accC, "
                          "coupling-OFF (per-message reject) vs coupling-ON (reject + neutralise)",
        "scenarios": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
