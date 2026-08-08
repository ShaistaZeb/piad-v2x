"""Figure data + plots for the multi-scale lifecycle coupling.

Companion to run_lifecycle_multiscale.py. Trains the SAME frozen multi-scale detector on
_1416 (SEED=42), scores the held-out _0709 persistent scenarios ONCE, then:

  1. reproduces the headline operating point (TAU=0.15, K=3) and checks it matches the
     locked lifecycle_multiscale.json (fails loudly if it drifts);
  2. sweeps a (TAU, K) grid -- cheap, reuses the one scoring pass -- to produce the
     reduction vs false-revocation operating curve;
  3. builds the Kaplan-Meier time-to-isolate curve at the operating point.

Outputs (experiments/results/):
  fig_tradeoff_multiscale.png   fig_tradeoff_multiscale.pdf
  fig_km_multiscale.png         fig_km_multiscale.pdf
  lifecycle_multiscale_figuredata.json   (the plotted numbers, for provenance)

Run from the piad-v2x repo root with the project venv.
"""
import glob, json, os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from piad_v2x.models.relational import relational_features, KINEMATIC, REL_TRUE
from piad_v2x.lifecycle import closed_loop as cl

SEED = 42
FEAT = KINEMATIC + REL_TRUE
ORACLE = {"sender", "senderPseudo", "is_attacker", "attackerType", "messageID"}
PERSISTENT_0709 = ["ConstPos_0709", "RandomPos_0709", "DataReplay_0709",
                   "Disruptive_0709", "DelayedMessages_0709", "DoS_0709"]
TARGET_RECALL = 0.90
TAU_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
K_GRID = [3, 5, 8]
OP_TAU, OP_K = 0.15, 3
RESDIR = "experiments/results"
FIGDIR = "experiments/figures"
os.makedirs(FIGDIR, exist_ok=True)
# Locked headline to guard against drift (from lifecycle_multiscale.json).
LOCKED = {"reduction": 0.956, "fr": 0.002, "frac_iso": 0.9959, "km": 1.0}


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
    paths = sorted(glob.glob("data/features/*_1416.parquet"))
    if not paths:
        raise SystemExit("no _1416 scenarios in data/features")
    df = pd.concat([with_relational(p) for p in paths], ignore_index=True)
    rng = np.random.default_rng(SEED)
    ug = np.array(df["sender"].astype(str).unique(), dtype=object); rng.shuffle(ug)
    val = set(ug[:int(0.2 * len(ug))])
    is_val = df["sender"].astype(str).isin(val).to_numpy()
    X = df[FEAT].fillna(0).to_numpy(); y = df["is_attacker"].to_numpy()
    Xtr, ytr = X[~is_val], y[~is_val]
    cap = 1_500_000
    if len(Xtr) > cap:
        idx = rng.choice(len(Xtr), size=cap, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    rf = RandomForestClassifier(n_estimators=120, max_depth=22, n_jobs=-1,
                                class_weight="balanced", random_state=SEED)
    rf.fit(Xtr, ytr)
    thr = pick_threshold(y[is_val], rf.predict_proba(X[is_val])[:, 1])
    print(f"trained multi-scale detector on _1416: {len(df):,} rows, "
          f"{len(ug)} senders, thr={thr:.4f}", flush=True)
    return rf, thr


def neutralisation(seqs, tau, K):
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


def km_curve(times, censored):
    """Return step points (t, S) of the Kaplan-Meier survival (still-active) curve."""
    order = np.argsort(times)
    t = np.asarray(times, float)[order]; c = np.asarray(censored)[order]
    n = len(t); at_risk = n; S = 1.0; i = 0
    ts, Ss = [0.0], [1.0]
    med = None
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]:
            if not c[j]:
                d += 1
            j += 1
        if d > 0 and at_risk > 0:
            S *= (1 - d / at_risk)
            ts.append(float(t[i])); Ss.append(S)
        if med is None and S <= 0.5:
            med = float(t[i])
        at_risk -= (j - i); i = j
    return np.array(ts), np.array(Ss), med


def outcomes(scored, tau, K):
    """Compute (reduction, false_revocation, frac_isolated, tti, tti_cens) at (tau,K)."""
    accC, accD = {}, {}
    benign_rev, benign_all = set(), set()
    tti, tti_cens = [], []
    for nm, df, seqs, maxt in scored:
        tneu = neutralisation(seqs, tau, K)
        atk = df[df.is_attacker == 1].copy()
        atk["t_neu"] = atk.senderPseudo.map(tneu)
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
    C = np.array([accC[k] for k in accC], float); D = np.array([accD[k] for k in accC], float)
    red = (C.sum() - D.sum()) / C.sum() if C.sum() > 0 else float("nan")
    fr = len(benign_rev) / max(1, len(benign_all))
    frac_iso = float(np.mean([1 - c for c in tti_cens])) if tti_cens else float("nan")
    return red, fr, frac_iso, np.array(tti), np.array(tti_cens)


def main():
    bad = [c for c in FEAT if c in ORACLE or c.startswith("true_")]
    if bad:
        raise SystemExit(f"oracle check failed: {bad}")
    rf, thr = train_multiscale_1416()

    scored = []
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
        scored.append((nm, df, seqs, float(df["rcvTime"].max())))
        print(f"[{nm:22s}] scored ({len(seqs)} pseudonyms)", flush=True)

    # ---- headline reproduction check ----
    red, fr, frac_iso, tti, tti_cens = outcomes(scored, OP_TAU, OP_K)
    kx, ky, kmed = km_curve(tti, tti_cens)
    print(f"\nOP (tau={OP_TAU}, K={OP_K}): reduction={red:.4f} fr={fr:.4f} "
          f"frac_iso={frac_iso:.4f} km_median={kmed}")
    drift = {"reduction": abs(red - LOCKED["reduction"]), "fr": abs(fr - LOCKED["fr"]),
             "frac_iso": abs(frac_iso - LOCKED["frac_iso"])}
    print("drift vs locked:", {k: round(v, 4) for k, v in drift.items()})
    if drift["reduction"] > 0.01 or drift["fr"] > 0.01 or drift["frac_iso"] > 0.01:
        raise SystemExit("HEADLINE DRIFT >0.01 vs locked JSON -- do not trust figures")

    # ---- (tau, K) sweep for the operating curve ----
    sweep = []
    for K in K_GRID:
        for tau in TAU_GRID:
            r, f, fi, _, _ = outcomes(scored, tau, K)
            sweep.append({"tau": tau, "K": K, "reduction": round(r, 4),
                          "false_revocation": round(f, 4), "frac_isolated": round(fi, 4)})
            print(f"  sweep tau={tau} K={K}: red={r:.4f} fr={f:.4f}", flush=True)

    os.makedirs(RESDIR, exist_ok=True)
    figdata = {
        "operating_point": {"tau": OP_TAU, "K": OP_K, "reduction": round(red, 4),
                            "false_revocation": round(fr, 4), "frac_isolated": round(frac_iso, 4),
                            "km_median_s": kmed},
        "sweep": sweep,
        "km_curve": {"t": [round(float(x), 3) for x in kx],
                     "fraction_active": [round(float(y), 4) for y in ky]},
        "n_attacker_pseudonyms": int(len(tti)),
    }
    json.dump(figdata, open(os.path.join(RESDIR, "lifecycle_multiscale_figuredata.json"), "w"),
              indent=2)

    # ================= FIGURE 1: operating curve =================
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    markers = {3: "o", 5: "s", 8: "^"}
    for K in K_GRID:
        pts = [s for s in sweep if s["K"] == K]
        xs = [s["false_revocation"] * 100 for s in pts]
        ys = [s["reduction"] * 100 for s in pts]
        ax.plot(xs, ys, marker=markers[K], ms=4, lw=1.2, label=f"K={K}")
    # highlight operating point
    ax.plot(fr * 100, red * 100, "*", color="crimson", ms=15, zorder=5,
            label=f"operating point\n($\\tau$={OP_TAU}, K={OP_K})")
    ax.set_xlabel("benign false-revocation (%)")
    ax.set_ylabel("accepted-malicious reduction (%)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Isolation-outcome operating curve", fontsize=9)
    fig.savefig(os.path.join(FIGDIR, "fig_tradeoff_multiscale.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_tradeoff_multiscale.pdf"))
    plt.close(fig)

    # ================= FIGURE 2: Kaplan-Meier =================
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.step(kx, ky * 100, where="post", lw=1.6, color="#1f4e79")
    if kmed is not None:
        ax.axhline(50, ls=":", color="grey", lw=1)
        ax.axvline(kmed, ls=":", color="crimson", lw=1)
        ax.plot([kmed], [50], "o", color="crimson", ms=6,
                label=f"median = {kmed:.0f}s")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("time since attacker's first message (s)")
    ax.set_ylabel("attacker pseudonyms\nnot yet isolated (%)")
    ax.set_ylim(-2, 102)
    xmax = float(np.percentile(kx, 99)) if len(kx) > 3 else float(kx.max())
    ax.set_xlim(0, max(5.0, xmax))
    ax.set_title("Time-to-isolate (Kaplan--Meier)", fontsize=9)
    fig.savefig(os.path.join(FIGDIR, "fig_km_multiscale.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_km_multiscale.pdf"))
    plt.close(fig)

    print("\nwrote figures + figuredata to", RESDIR)


if __name__ == "__main__":
    main()
