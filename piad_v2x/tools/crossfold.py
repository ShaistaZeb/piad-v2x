"""Cross-fold + multi-seed evaluation for PIAD-V2X (repeated experiments).

Determines whether the RandomForest's observed lead over the PINN variants is
stable, by repeating the full train -> calibrate -> evaluate cycle across
several seeds and all cross-validation folds, then aggregating and running
paired statistical tests.

Per run (seed, fold):
  * train RF + PINN-kin + PINN-full on the leakage-safe unseen-attack split
    (identical split for all three -> a true paired comparison),
  * choose thresholds on a sender-disjoint CALIBRATION slice of the test set,
  * report metrics on the disjoint EVALUATION slice.

Records per (run, model): AUC, plus precision/recall/F1/accuracy at the primary
operating point (5% FPR), the selected threshold, and recall at 1/5/10% FPR.
Aggregates mean / std / 95% CI, and runs paired t-tests + Wilcoxon signed-rank
for RF-vs-kin, RF-vs-full, kin-vs-full (Bonferroni-corrected for 3 comparisons).

No claim of superiority is written unless the repeated tests support it. The
saved models from train.py are NOT touched; this script trains its own
throwaway per-run models.

Usage:
    python piad_v2x/tools/crossfold.py --data data/veremi.parquet --rows 200000 \
        --seeds 17 23 42 --n-splits 5 --epochs 6 --out checkpoints/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from piad_v2x.models.baseline_kin import BaselineConfig, train_baseline
from piad_v2x.data_utils.dataset import DatasetSpec, load_messages, split_holdout_attacks
from piad_v2x.models.detector import (
    DetectorConfig, evaluate_detector, predict_proba, train_detector,
)
from piad_v2x.data_utils.feature_extractor import extract_features

MODELS = ["Random Forest", "PINN-kin", "PINN-full"]
PAIRS = [("Random Forest", "PINN-kin"), ("Random Forest", "PINN-full"),
         ("PINN-kin", "PINN-full")]
TEST_METRICS = ["auc", "recall_fpr5", "f1"]   # metrics carried into the paired tests


def fnum(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.4f}" if x == x else "nan"


def op_at_threshold(y_bin, p_attack, thr):
    pred = (p_attack >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_bin, pred, average="binary", pos_label=1, zero_division=0)
    benign = y_bin == 0
    fpr = float(pred[benign].mean()) if benign.any() else float("nan")
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "accuracy": float((pred == y_bin).mean()), "fpr": fpr}


def score_model(kind, train_X, train_y, send_time, test_X, test_y, cfg_seed, epochs, device):
    """Train one model and return its per-test-row (pred_class, p_attack)."""
    if kind == "rf":
        rf = train_baseline(train_X, train_y, BaselineConfig(random_state=cfg_seed))
        pred = rf.predict(test_X)
        zc = list(rf.classes_).index(0) if 0 in rf.classes_ else None
        p_attack = (1.0 - rf.predict_proba(test_X)[:, zc]) if zc is not None \
            else rf.predict_proba(test_X).sum(axis=1)
        del rf
        return pred, p_attack
    lam = 0.0 if kind == "pinn_kin" else 0.5
    cfg = DetectorConfig(epochs=epochs, lambda_kin=0.5, lambda_lwr=lam, seed=cfg_seed)
    model, _ = train_detector(train_X, train_y, send_time, cfg=cfg, device=device)
    pred = evaluate_detector(model, test_X, test_y, device=device)
    p_attack = 1.0 - predict_proba(model, test_X, device=device)[:, 0]
    del model
    return pred, p_attack


def run_once(df, spec, holdout, fprs, cal_frac, epochs, device, split_seed):
    """One (seed, fold) run -> {model: metric dict}. Identical split for all models."""
    train_df, test_df = split_holdout_attacks(df, holdout, spec)
    ftr = extract_features(train_df)
    send_time = train_df["sendTime"].to_numpy(dtype="float64")
    fte = extract_features(test_df)
    test_X, test_y = fte.X, fte.y
    y_bin = (test_y != 0).astype(int)

    # sender-disjoint calibration / evaluation slices
    rng = np.random.default_rng(split_seed)
    uniq = np.unique(fte.senders)
    perm = rng.permutation(uniq)
    n_cal = max(1, int(len(uniq) * cal_frac))
    cal = np.isin(fte.senders, list(perm[:n_cal].tolist()))
    ev = ~cal
    if not (y_bin[cal] == 0).any() or not (y_bin[cal] == 1).any():
        raise RuntimeError("calibration slice missing a class")
    if not (y_bin[ev] == 0).any() or not (y_bin[ev] == 1).any():
        raise RuntimeError("evaluation slice missing a class")

    out = {}
    for kind, name in (("rf", "Random Forest"), ("pinn_kin", "PINN-kin"),
                       ("pinn_full", "PINN-full")):
        pred, p_attack = score_model(kind, ftr.X, ftr.y, send_time, test_X, test_y,
                                     spec.random_state, epochs, device)
        cal_benign = p_attack[cal & (y_bin == 0)]
        rec = {"auc": float(roc_auc_score(y_bin[ev], p_attack[ev]))}
        primary_thr = None
        for tf in fprs:
            thr = float(np.quantile(cal_benign, 1.0 - tf))
            m = op_at_threshold(y_bin[ev], p_attack[ev], thr)
            rec[f"recall_fpr{int(tf*100)}"] = m["recall"]
            if abs(tf - 0.05) < 1e-9:  # primary operating point = 5% FPR
                primary_thr = thr
                rec.update({"threshold": thr, "precision": m["precision"],
                            "recall": m["recall"], "f1": m["f1"],
                            "accuracy": m["accuracy"], "fpr_eval": m["fpr"]})
        out[name] = rec
    return out


def agg(values):
    a = np.asarray([v for v in values if v == v], dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    else:
        h = float("nan")
    return {"mean": mean, "std": std, "ci95_low": mean - h, "ci95_high": mean + h, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-fold + multi-seed evaluation.")
    ap.add_argument("--data", type=Path, default=Path("data/veremi.parquet"))
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--holdout", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    ap.add_argument("--fprs", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"crossfold.py: dataset not found: {args.data}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    holdout = tuple(int(c) for c in args.holdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fprs = sorted(args.fprs)

    runs = []  # list of dicts: seed, fold, model, metrics...
    t0 = time.perf_counter()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        df = load_messages(DatasetSpec(csv_path=str(args.data),
                                       n_rows=None if args.rows == 0 else args.rows,
                                       random_state=seed, n_splits=args.n_splits))
        for fold in range(args.n_splits):
            spec = DatasetSpec(csv_path=str(args.data),
                               n_rows=None if args.rows == 0 else args.rows,
                               random_state=seed, n_splits=args.n_splits, fold_index=fold)
            try:
                res = run_once(df, spec, holdout, fprs, args.cal_frac,
                               args.epochs, device, split_seed=seed * 1000 + fold)
            except Exception as e:  # one bad fold must not kill the sweep
                print(f"  seed={seed} fold={fold} SKIPPED: {e}")
                continue
            for name, rec in res.items():
                runs.append({"seed": seed, "fold": fold, "model": name, **rec})
            print(f"  seed={seed} fold={fold} done "
                  f"(AUC RF={res['Random Forest']['auc']:.3f} "
                  f"kin={res['PINN-kin']['auc']:.3f} full={res['PINN-full']['auc']:.3f})")
    elapsed = time.perf_counter() - t0
    n_runs = len({(r["seed"], r["fold"]) for r in runs})
    print(f"\ncompleted {n_runs} runs in {elapsed:.0f}s")
    if n_runs < 2:
        sys.exit("crossfold.py: fewer than 2 successful runs; cannot aggregate.")

    # ---------- per-run CSV ----------
    metric_cols = ["auc", "threshold", "precision", "recall", "f1", "accuracy",
                   "recall_fpr1", "recall_fpr5", "recall_fpr10"]
    csv_path = out / "results_crossfold.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "fold", "model"] + metric_cols)
        for r in runs:
            w.writerow([r["seed"], r["fold"], r["model"]] + [r.get(c, "") for c in metric_cols])

    # ---------- aggregate ----------
    by_model = {m: [r for r in runs if r["model"] == m] for m in MODELS}
    aggregate = {m: {c: agg([r.get(c) for r in by_model[m]]) for c in metric_cols}
                 for m in MODELS}

    # ---------- paired tests (same seeds+folds) ----------
    def paired_vector(model, metric):
        d = {(r["seed"], r["fold"]): r.get(metric) for r in by_model[model]}
        return d

    alpha, bonf = 0.05, 0.05 / len(PAIRS)
    paired = {}
    for metric in TEST_METRICS:
        paired[metric] = {}
        for a, b in PAIRS:
            va, vb = paired_vector(a, metric), paired_vector(b, metric)
            keys = sorted(set(va) & set(vb))
            xa = np.array([va[k] for k in keys], float)
            xb = np.array([vb[k] for k in keys], float)
            md = float(np.mean(xa - xb))
            try:
                t_stat, p_t = stats.ttest_rel(xa, xb)
                t_stat, p_t = float(t_stat), float(p_t)
            except Exception:
                t_stat, p_t = float("nan"), float("nan")
            try:
                _, p_w = stats.wilcoxon(xa, xb)
                p_w = float(p_w)
            except Exception:
                p_w = float("nan")
            paired[metric][f"{a} vs {b}"] = {
                "n": len(keys), "mean_diff": md, "t_stat": t_stat,
                "p_ttest": p_t, "p_wilcoxon": p_w,
                "sig_raw": bool(p_t == p_t and p_t < alpha),
                "sig_bonferroni": bool(p_t == p_t and p_t < bonf),
            }

    statistics = {
        "config": {"seeds": args.seeds, "n_splits": args.n_splits, "n_runs": n_runs,
                   "rows": args.rows, "holdout": list(holdout), "epochs": args.epochs,
                   "primary_operating_point": "FPR=5%", "cal_frac": args.cal_frac,
                   "device": str(device), "elapsed_seconds": round(elapsed)},
        "aggregate": aggregate,
        "paired_tests": paired,
        "alpha": alpha, "bonferroni_alpha": bonf, "n_comparisons": len(PAIRS),
    }
    (out / "statistics.json").write_text(json.dumps(statistics, indent=2))

    # ---------- summary.md (Observed / Interpretation / Limitations) ----------
    def row_ci(m, metric):
        a = aggregate[m][metric]
        return f"{fnum(a['mean'])} ± {fnum(a['std'])} [{fnum(a['ci95_low'])}, {fnum(a['ci95_high'])}]"

    L = ["# PIAD-V2X Cross-Fold / Multi-Seed Evaluation", "",
         f"Seeds: {args.seeds}  |  folds: {args.n_splits}  |  successful runs: {n_runs}  |  "
         f"rows: {args.rows:,}  |  holdout: {list(holdout)}  |  PINN epochs: {args.epochs}  |  "
         f"primary operating point: 5% FPR  |  device: {device}", "",
         "## Observed evidence", "",
         "Mean ± SD [95% CI] across all runs (threshold chosen on calibration slice, "
         "metrics on disjoint evaluation slice).", "",
         "| Model | AUC | Recall @5% FPR | F1 @5% FPR | Accuracy @5% FPR | Recall @1% | Recall @10% |",
         "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        L.append(f"| {m} | {row_ci(m,'auc')} | {row_ci(m,'recall_fpr5')} | {row_ci(m,'f1')} | "
                 f"{row_ci(m,'accuracy')} | {row_ci(m,'recall_fpr1')} | {row_ci(m,'recall_fpr10')} |")
    L += ["", "### Paired statistical tests (same seeds and folds)", "",
          f"Paired t-test and Wilcoxon signed-rank, n={n_runs} per pair. "
          f"Bonferroni alpha for {len(PAIRS)} comparisons = {bonf:.4f}.", "",
          "| Metric | Comparison | Mean diff (A-B) | t p-value | Wilcoxon p | Significant (Bonferroni) |",
          "|---|---|---|---|---|---|"]
    for metric in TEST_METRICS:
        for pair, d in paired[metric].items():
            L.append(f"| {metric} | {pair} | {fnum(d['mean_diff'])} | {fnum(d['p_ttest'])} | "
                     f"{fnum(d['p_wilcoxon'])} | {'YES' if d['sig_bonferroni'] else 'no'} |")

    # interpretation strictly from the tests
    L += ["", "## Interpretation", ""]
    auc = paired["auc"]
    def verdict(pair):
        d = auc[pair]
        a, b = pair.split(" vs ")
        if d["mean_diff"] != d["mean_diff"]:
            return f"- {pair} (AUC): not computable."
        leader, lag = (a, b) if d["mean_diff"] > 0 else (b, a)
        if d["sig_bonferroni"]:
            return (f"- {pair} (AUC): {leader} higher by {abs(d['mean_diff']):.4f}, "
                    f"p={fnum(d['p_ttest'])} < {bonf:.4f}. Difference is statistically "
                    f"significant across the repeated runs.")
        return (f"- {pair} (AUC): {leader} higher by {abs(d['mean_diff']):.4f}, but "
                f"p={fnum(d['p_ttest'])} is NOT significant after Bonferroni; "
                f"no superiority claim is warranted.")
    for pair in [f"{a} vs {b}" for a, b in PAIRS]:
        L.append(verdict(pair))
    L += ["",
          "Superiority is asserted only where a paired test is significant after correction; "
          "elsewhere the wording is deliberately 'no difference established'.", "",
          "## Limitations", "",
          "- The 5 folds within each seed share the same subsampled pool, so per-run results "
          "are not fully independent. The paired tests are therefore approximate and likely "
          "anti-conservative (real p-values may be larger).",
          f"- Only {len(args.seeds)} seeds x {args.n_splits} folds; a larger grid would tighten the CIs.",
          "- One dataset (VeReMi Extension), one held-out attack family set "
          f"({list(holdout)}), one feature set; results may not transfer.",
          "- Operating points use a fixed 5% (and 1/10%) benign FPR target; a different "
          "deployment FPR would shift the recall/precision trade-off.",
          "- CPU training, fixed PINN hyperparameters (lambda, epochs); a PINN hyperparameter "
          "search was not performed, so 'PINN does not beat RF' is conditional on these settings.",
          ""]
    (out / "results_summary.md").write_text("\n".join(L))
    print(f"wrote results_crossfold.csv / statistics.json / results_summary.md to {out}/")


if __name__ == "__main__":
    main()
