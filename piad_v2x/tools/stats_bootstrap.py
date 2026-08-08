#!/usr/bin/env python3
"""
Cluster bootstrap over SENDERS for the placement-study family (pre-registered).

Resamples sender groups (gid) with replacement, recomputes macro-F1 on the hard
classes for each arm, and reports paired differences (Arm iii - Arm i) and
(Arm iii - Arm ii) with 95% CIs and Holm-Bonferroni. Efficient: per-gid, per-arm
confusion counts are precomputed, so each bootstrap replicate is a sum over
sampled gids, not a re-scan of millions of rows.
"""
import json, os
import numpy as np

RES = "experiments/results"
HARD = ["DataReplay_1416", "DataReplaySybil_1416", "Disruptive_1416", "EventualStop_1416"]
MDE = 0.02
NREP = 1000
SEED = 42


def load():
    d = np.load(os.path.join(RES, "test_preds.npz"), allow_pickle=True)
    return d


def per_gid_counts(y, gid, attack_class, prob, thr):
    """Per sender group, record benign flag counts AND per-attack-class attacker
    flag counts. A sender under bare-sender grouping spans multiple scenarios, so
    it can contribute attacker messages to several classes; benign senders
    contribute only benign counts. Returns gid -> dict."""
    yhat = (prob >= thr).astype(int)
    import pandas as pd
    df = pd.DataFrame({"gid": gid, "y": y, "cls": attack_class, "yhat": yhat})
    counts = {}
    for g, sub in df.groupby("gid"):
        benign = sub[sub["y"] == 0]; atk = sub[sub["y"] == 1]
        d = {"benign_flagged": int(benign["yhat"].sum()),
             "benign_total": int(len(benign)), "atk_by_class": {}}
        for c, cc in atk.groupby("cls"):
            d["atk_by_class"][c] = [int(cc["yhat"].sum()), int(len(cc))]
        counts[g] = d
    return counts


def macro_f1_hard(sampled_gids, counts):
    # pooled benign false positives (all benign are negatives for every class);
    # per hard class, sum attacker TP/total across sampled senders that have them.
    fp_benign = sum(counts[g]["benign_flagged"] for g in sampled_gids)
    per_class = {}
    for g in sampled_gids:
        for c, (fl, tot) in counts[g]["atk_by_class"].items():
            if c in HARD:
                acc = per_class.setdefault(c, [0, 0]); acc[0] += fl; acc[1] += tot
    f1s = []
    for c in HARD:
        if c not in per_class:
            continue
        tp, tot = per_class[c]; fn = tot - tp
        denom = 2 * tp + fp_benign + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    d = load()
    y = d["y"]; gid = d["gid"].astype(str); ac = d["attack_class"].astype(str)
    arms = {"arm_i": (d["arm_i"], float(d["thr_i"])),
            "arm_ii": (d["arm_ii"], float(d["thr_ii"])),
            "arm_iii": (d["arm_iii"], float(d["thr_iii"]))}
    counts = {a: per_gid_counts(y, gid, ac, p, t) for a, (p, t) in arms.items()}
    uniq = np.unique(gid); n = len(uniq)
    rng = np.random.default_rng(SEED)

    point = {a: macro_f1_hard(uniq, counts[a]) for a in arms}
    diffs = {"iii_minus_i": [], "iii_minus_ii": []}
    for _ in range(NREP):
        samp = rng.choice(uniq, size=n, replace=True)
        f_i = macro_f1_hard(samp, counts["arm_i"])
        f_ii = macro_f1_hard(samp, counts["arm_ii"])
        f_iii = macro_f1_hard(samp, counts["arm_iii"])
        diffs["iii_minus_i"].append(f_iii - f_i)
        diffs["iii_minus_ii"].append(f_iii - f_ii)

    def summ(vals):
        v = np.array(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        # two-sided bootstrap p for H0: diff=0
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        return dict(mean=round(float(v.mean()), 4), ci_lo=round(float(lo), 4),
                    ci_hi=round(float(hi), 4), p=round(float(min(1.0, p)), 4))

    out = {"point_macro_f1_hard": {a: round(point[a], 4) for a in arms},
           "MDE": MDE,
           "H1a_iii_vs_i": summ(diffs["iii_minus_i"]),
           "H1b_iii_vs_ii": summ(diffs["iii_minus_ii"])}
    # Holm-Bonferroni over the 2 family tests
    tests = [("H1a_iii_vs_i", out["H1a_iii_vs_i"]["p"]),
             ("H1b_iii_vs_ii", out["H1b_iii_vs_ii"]["p"])]
    tests.sort(key=lambda x: x[1])
    m = len(tests); holm = {}
    for i, (name, p) in enumerate(tests):
        holm[name] = round(min(1.0, p * (m - i)), 4)
    out["holm_adjusted_p"] = holm
    # verdicts
    def verdict(diff, adj_p):
        meaningful = diff["mean"] >= MDE and diff["ci_lo"] > 0 and adj_p < 0.05
        return "supported" if meaningful else (
            "refuted (indistinguishable / below MDE)" if abs(diff["mean"]) < MDE or diff["ci_lo"] <= 0 <= diff["ci_hi"]
            else "not supported")
    out["verdict_H1a"] = verdict(out["H1a_iii_vs_i"], holm["H1a_iii_vs_i"])
    out["verdict_H1b"] = verdict(out["H1b_iii_vs_ii"], holm["H1b_iii_vs_ii"])
    out["n_sender_groups"] = int(n)

    with open(os.path.join(RES, "placement_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote", os.path.join(RES, "placement_bootstrap.json"))


if __name__ == "__main__":
    main()
