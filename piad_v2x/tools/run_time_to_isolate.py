"""Time-to-isolate: the second isolation-outcome metric of the trust-gated lifecycle.

Completes the outcome-metric pair. run_operating_point.py delivered the first metric
(accepted-malicious-message reduction). This delivers time-to-isolate: for each attacker
pseudonym, the wall-clock from its first malicious message to neutralisation, at the
pre-registered operating point (TAU_REV=0.15, K=3), on the held-out `_0709` window.

Four guards (see the run summary; each is asserted or reported, never assumed):
  G1  no train-on-test: `_0709`, detector frozen on `_1416`, every sender unseen.
  G2  censoring: attacker pseudonyms never neutralised are RIGHT-CENSORED. Median is
      Kaplan-Meier (not a survivorship-biased mean over the isolated only); the fraction
      ever isolated is reported alongside.
  G3  honest clock + population: clock starts at each attacker pseudonym's FIRST malicious
      message; only attacker pseudonyms; pseudonyms shorter than the N_MIN cold-start can
      never be neutralised and are reported as STRUCTURALLY censored (the short-lived /
      Sybil boundary, shown not hidden).
  #7  oracle check: the scored feature set must contain NO oracle columns
      (sender, senderPseudo, is_attacker, attackerType, messageID, true_*). Neutralisation
      is driven only by the detector's p_mal, never by labels. VOID if violated.

Run from the repo root:
    python piad_v2x/tools/run_time_to_isolate.py --out experiments/results/time_to_isolate.json
"""
import argparse, json, os
import numpy as np
import pandas as pd
import joblib

from piad_v2x.lifecycle import closed_loop as cl

PERSISTENT_0709 = ["ConstPos_0709", "RandomPos_0709", "DataReplay_0709",
                   "Disruptive_0709", "DelayedMessages_0709", "DoS_0709"]
TAU, K = 0.15, 3                       # pre-registered operating point
ORACLE = {"sender", "senderPseudo", "is_attacker", "attackerType", "messageID"}


def oracle_check(sc):
    """#7: the scored feature columns must contain no oracle/label columns."""
    feats = list(sc["KINEMATIC"]) + list(sc["RESIDUALS"])
    bad = [c for c in feats if c in ORACLE or c.startswith("true_")]
    return feats, bad


def isolate_times(df, tau, K):
    """Per senderPseudo: (t_neu, t0_first_msg, n_msgs). t_neu=inf if never neutralised.
    Reuses closed_loop's ALPHA/N_MIN/T0; only TAU/K are set by the operating point."""
    out = {}
    for pseudo, g in df.sort_values("rcvTime").groupby("senderPseudo"):
        T = cl.T0; below = 0; n = 0; t_neu = np.inf
        rcv = g["rcvTime"].to_numpy()
        for b, t in zip(1.0 - g["p_mal"].to_numpy(), rcv):
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
        out[pseudo] = (t_neu, float(rcv[0]), int(len(rcv)))
    return out


def km_median(times, censored):
    """Kaplan-Meier median of time-to-event with right-censoring.
    times: event/censor times; censored[i]=True means right-censored (no event)."""
    order = np.argsort(times)
    t = np.asarray(times)[order]; c = np.asarray(censored)[order]
    n = len(t); at_risk = n; S = 1.0
    med = None
    i = 0
    while i < n:
        # group ties at the same time
        j = i
        d = 0  # events at this time
        while j < n and t[j] == t[i]:
            if not c[j]:
                d += 1
            j += 1
        if d > 0 and at_risk > 0:
            S *= (1 - d / at_risk)
        if med is None and S <= 0.5:
            med = float(t[i])
        at_risk -= (j - i)
        i = j
    return med, float(S)  # median (None if survival never reached 0.5), final survival


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=PERSISTENT_0709)
    ap.add_argument("--out", default="experiments/results/time_to_isolate.json")
    args = ap.parse_args()

    rf = joblib.load(os.path.join(cl.MODELS, "rf.joblib"))
    sc = joblib.load(os.path.join(cl.MODELS, "scalers.joblib"))

    feats, bad = oracle_check(sc)
    if bad:
        raise SystemExit(f"#7 ORACLE CHECK FAILED: scored features include oracle columns "
                         f"{bad}. Run is VOID.")
    print(f"#7 oracle check PASS: {len(feats)} scored features, none oracle "
          f"({', '.join(feats[:6])}...).", flush=True)

    per_scen = {}
    all_tti = []          # finite time-to-isolate values (isolated pseudonyms)
    all_cens = []         # 1 = censored (never isolated), 0 = event
    n_attacker = 0; n_isolated = 0; n_short = 0
    for nm in args.scenarios:
        p = os.path.join("data/features", nm + ".parquet")
        if not os.path.exists(p):
            print("skip missing", nm); continue
        df = pd.read_parquet(p)
        df = df[df.attackerType >= 0].copy()
        df["p_mal"], _ = cl.score(df, rf, sc)             # detector output only
        atk = df[df.is_attacker == 1]                     # G3: attacker pseudonyms only
        it = isolate_times(atk, TAU, K)
        tti = []; cens = 0; short = 0
        for pseudo, (t_neu, t0, nmsg) in it.items():
            n_attacker += 1
            if nmsg < cl.N_MIN:                            # G3: structurally censored
                short += 1
            if np.isfinite(t_neu):
                dt = max(0.0, t_neu - t0)
                tti.append(dt); all_tti.append(dt); all_cens.append(0)
                n_isolated += 1
            else:                                          # G2: right-censored
                cens += 1
                all_tti.append(float(df["rcvTime"].max() - t0)); all_cens.append(1)
        n_short += short
        # per-scenario KM over this scenario's attacker pseudonyms (events + censored)
        scen_censored_times = [df["rcvTime"].max() - it[p2][1]
                               for p2 in it if not np.isfinite(it[p2][0])]
        if it:
            scen_med, _ = km_median(tti + scen_censored_times,
                                    [0] * len(tti) + [1] * len(scen_censored_times))
        else:
            scen_med = None
        frac_iso = len(tti) / max(1, len(it))
        per_scen[nm] = {
            "n_attacker_pseudonyms": len(it),
            "n_isolated": len(tti),
            "frac_isolated": round(frac_iso, 4),
            "n_structurally_censored_short": short,
            "km_median_time_to_isolate_s": round(scen_med, 3) if scen_med is not None else None,
            "median_of_isolated_only_s": round(float(np.median(tti)), 3) if tti else None,
        }
        print(f"[{nm:22s}] attacker_pseudos={len(it):5d} isolated={len(tti):5d} "
              f"({frac_iso:.2%})  KM median TTI={per_scen[nm]['km_median_time_to_isolate_s']}s "
              f"short-censored={short}", flush=True)

    pooled_med, _ = km_median(all_tti, all_cens) if all_tti else (None, 1.0)
    summary = {
        "protocol": "closed-loop isolation outcomes; second outcome metric",
        "window": "_0709 (held-out; detector frozen on _1416, all senders unseen) [G1]",
        "operating_point": {"TAU_REV": TAU, "K": K},
        "guards": {
            "G1_no_train_on_test": "detector frozen on _1416; every _0709 sender unseen",
            "G2_censoring": "never-isolated pseudonyms right-censored; median is Kaplan-Meier",
            "G3_clock_and_population": "clock from first malicious message; attacker pseudonyms "
                                       "only; sub-N_MIN pseudonyms structurally censored",
            "oracle_check_7": f"PASS - {len(feats)} scored features, none oracle/label",
        },
        "pooled": {
            "n_attacker_pseudonyms": n_attacker,
            "n_isolated": n_isolated,
            "frac_isolated": round(n_isolated / max(1, n_attacker), 4),
            "n_structurally_censored_short_lived": n_short,
            "km_median_time_to_isolate_s": round(pooled_med, 3) if pooled_med is not None else None,
        },
        "per_scenario": per_scen,
        "note": "Time-to-isolate pairs with accepted-malicious-message reduction "
                "(operating_point) as the two isolation-outcome metrics. "
                "Short-lived (sub-cold-start) attacker pseudonyms are never isolated by "
                "design - the honest Sybil/short-pseudonym boundary.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\npooled: {n_isolated}/{n_attacker} attacker pseudonyms isolated "
          f"({summary['pooled']['frac_isolated']:.2%}), KM median TTI="
          f"{summary['pooled']['km_median_time_to_isolate_s']}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
