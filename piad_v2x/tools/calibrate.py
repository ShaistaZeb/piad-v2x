"""Threshold calibration for PIAD-V2X detectors (evaluation only).

Motivation: the default decision rule (argmax over the 20-way class head, then
"non-zero = misbehaviour") yields a degenerate operating point that flags almost
all traffic as attack. AUC shows the per-message attack score `p_attack` ranks
attacks above benign well; the problem is the operating point, not necessarily
the model. A deployed detector would instead threshold `p_attack` at a tolerable
false-positive rate. This script does exactly that, without retraining.

Rigor: the threshold is chosen on a CALIBRATION slice of the held-out test set
and the metrics are reported on a disjoint EVALUATION slice, split BY SENDER so
no vehicle straddles the two. Choosing the threshold on the same rows it is
scored on would be circular; this avoids that.

It never trains, never edits the models, and treats manifest.json as the single
source of truth. Writes calibration.{json,csv,txt} into the checkpoint dir.

Usage:
    python piad_v2x/tools/calibrate.py --ckpt checkpoints/
    python piad_v2x/tools/calibrate.py --ckpt checkpoints/ --fpr 0.01 0.05 0.10 --cal-frac 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import joblib
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from piad_v2x.data_utils.dataset import DatasetSpec, load_messages, split_holdout_attacks
from piad_v2x.models.detector import DetectorConfig, PINNDetector, evaluate_detector, predict_proba
from piad_v2x.data_utils.feature_extractor import extract_features

MODEL_ORDER = [("rf", "Random Forest"), ("pinn_kin", "PINN-kin"), ("pinn_full", "PINN-full")]


def die(msg: str) -> None:
    sys.exit(f"calibrate.py: {msg}")


def f(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.4f}" if x == x else "nan"


def render(headers, rows):
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = lambda c: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(c))
    return "\n".join([line(headers), "  ".join("-" * x for x in w), *(line(r) for r in rows)])


def op_metrics(y_bin: np.ndarray, p_attack: np.ndarray, thr: float) -> dict:
    """Binary metrics at a fixed score threshold."""
    pred = (p_attack >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_bin, pred, average="binary", pos_label=1, zero_division=0)
    acc = float((pred == y_bin).mean())
    benign = y_bin == 0
    fpr = float(pred[benign].mean()) if benign.any() else float("nan")
    return {"threshold": float(thr), "fpr": fpr, "precision": float(p),
            "recall": float(r), "f1": float(f1), "accuracy": acc}


def model_scores(ckpt: Path, manifest: dict, X, y, device):
    """Return {display_name: (pred_class, p_attack)} for the saved models."""
    out = {}
    for key, name in MODEL_ORDER:
        meta = manifest["models"].get(key)
        if meta is None:
            die(f"manifest has no entry for '{key}'.")
        mfile = ckpt / meta["file"]
        if not mfile.exists():
            die(f"missing model file: {mfile}.")
        if meta["type"] == "RandomForest":
            rf = joblib.load(mfile)
            pred = rf.predict(X)
            zc = list(rf.classes_).index(0) if 0 in rf.classes_ else None
            p_attack = (1.0 - rf.predict_proba(X)[:, zc]) if zc is not None \
                else rf.predict_proba(X).sum(axis=1)
        elif meta["type"] == "PINN":
            cfg = DetectorConfig(**meta["config"])
            model = PINNDetector(cfg).to(device)
            model.load_state_dict(torch.load(mfile, map_location=device))
            model.eval()
            pred = evaluate_detector(model, X, y, device=device)
            p_attack = 1.0 - predict_proba(model, X, device=device)[:, 0]
        else:
            die(f"unknown model type '{meta['type']}'.")
        out[name] = (pred, p_attack)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold calibration (evaluation only).")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    ap.add_argument("--data", type=Path, default=None, help="override CSV path only")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                    help="target benign false-positive rates for operating points")
    ap.add_argument("--cal-frac", type=float, default=0.5,
                    help="fraction of test senders used to choose the threshold")
    args = ap.parse_args()

    ckpt = args.ckpt
    man = ckpt / "manifest.json"
    if not man.exists():
        die(f"manifest.json not found in {ckpt}/ . Run piad_v2x/tools/train.py + inference.py first.")
    manifest = json.loads(man.read_text())

    data_path = Path(args.data) if args.data else Path(manifest["data"])
    if not data_path.exists():
        die(f"dataset not found: {data_path}\nRun from repo root or pass --data.")

    seed = manifest["seed"]
    holdout = tuple(int(c) for c in manifest["holdout"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rebuild the identical held-out test set
    spec = DatasetSpec(csv_path=str(data_path),
                       n_rows=None if manifest["rows"] == 0 else manifest["rows"],
                       fold_index=manifest["fold"], random_state=seed)
    df = load_messages(spec)
    _, test_df = split_holdout_attacks(df, holdout, spec)
    feats = extract_features(test_df)
    X, y = feats.X, feats.y
    y_bin = (y != 0).astype(int)
    senders = feats.senders

    # disjoint calibration / evaluation slices, split BY SENDER
    rng = np.random.default_rng(seed)
    uniq = np.unique(senders)
    perm = rng.permutation(uniq)
    n_cal = max(1, int(len(uniq) * args.cal_frac))
    cal_senders = set(perm[:n_cal].tolist())
    cal = np.isin(senders, list(cal_senders))
    ev = ~cal
    for nm, mask in (("calibration", cal), ("evaluation", ev)):
        nb, na = int((y_bin[mask] == 0).sum()), int((y_bin[mask] == 1).sum())
        if nb == 0 or na == 0:
            die(f"{nm} slice has benign={nb} attack={na}; need both. Adjust --cal-frac.")
    print(f"test={len(test_df):,}  calibration={int(cal.sum()):,}  evaluation={int(ev.sum()):,} "
          f"(sender-disjoint)  holdout={holdout}")

    scores = model_scores(ckpt, manifest, X, y, device)

    results: dict[str, dict] = {}
    for name, (pred_class, p_attack) in scores.items():
        cal_benign = p_attack[cal & (y_bin == 0)]
        auc_ev = float(roc_auc_score(y_bin[ev], p_attack[ev])) if y_bin[ev].any() else float("nan")
        ops: dict[str, dict] = {}

        # default argmax operating point (the degenerate baseline), on eval slice
        pred_default = (pred_class[ev] != 0).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_bin[ev], pred_default, average="binary", pos_label=1, zero_division=0)
        ops["default-argmax"] = {
            "threshold": None, "fpr": float(pred_default[y_bin[ev] == 0].mean()),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "accuracy": float((pred_default == y_bin[ev]).mean())}

        # target-FPR operating points: threshold = benign-score quantile on calibration
        for tf in args.fpr:
            thr = float(np.quantile(cal_benign, 1.0 - tf))
            ops[f"FPR={tf:.0%}"] = op_metrics(y_bin[ev], p_attack[ev], thr)

        # max-F1 threshold chosen on calibration, reported on evaluation
        cand = np.quantile(p_attack[cal], np.linspace(0.01, 0.99, 99))
        best_thr, best_f1 = cand[0], -1.0
        for thr in cand:
            pc = (p_attack[cal] >= thr).astype(int)
            _, _, f1c, _ = precision_recall_fscore_support(
                y_bin[cal], pc, average="binary", pos_label=1, zero_division=0)
            if f1c > best_f1:
                best_f1, best_thr = f1c, float(thr)
        ops["max-F1 (cal)"] = op_metrics(y_bin[ev], p_attack[ev], best_thr)

        results[name] = {"auc_eval": auc_ev, "operating_points": ops}

    # ---------- tables ----------
    op_names = ["default-argmax"] + [f"FPR={tf:.0%}" for tf in args.fpr] + ["max-F1 (cal)"]
    headers = ["Model", "Operating point", "Threshold", "FPR(eval)", "Precision", "Recall", "F1", "Accuracy"]
    rows = []
    for name in (n for _, n in MODEL_ORDER):
        for op in op_names:
            d = results[name]["operating_points"][op]
            thr = "argmax" if d["threshold"] is None else f"{d['threshold']:.4f}"
            rows.append([name, op, thr, f(d["fpr"]), f(d["precision"]),
                         f(d["recall"]), f(d["f1"]), f(d["accuracy"])])

    auc_rows = [[n, f(results[n]["auc_eval"])] for _, n in MODEL_ORDER]

    # honest comparison at a deployment-relevant matched FPR (prefer 5%, else first target)
    match_fpr = "FPR=5%" if "FPR=5%" in op_names else f"FPR={args.fpr[0]:.0%}"
    recall_at = {n: results[n]["operating_points"][match_fpr]["recall"] for _, n in MODEL_ORDER}
    best_recall_model = max(recall_at, key=recall_at.get)

    interp = []
    interp.append(f"AUC (threshold-independent, evaluation slice): "
                  + ", ".join(f"{n}={f(results[n]['auc_eval'])}" for _, n in MODEL_ORDER) + ".")
    interp.append(f"At a matched {match_fpr} operating point, recall (true-positive rate): "
                  + ", ".join(f"{n}={f(recall_at[n])}" for _, n in MODEL_ORDER)
                  + f"  -> highest: {best_recall_model}.")
    # default vs calibrated accuracy, to separate "bad operating point" from "bad model"
    for _, n in MODEL_ORDER:
        d_def = results[n]["operating_points"]["default-argmax"]["accuracy"]
        d_cal = results[n]["operating_points"][match_fpr]["accuracy"]
        interp.append(f"{n}: accuracy default-argmax={f(d_def)} vs {match_fpr}={f(d_cal)} "
                      f"(the model ranks the same; only the operating point changed).")
    interp.append("Threshold was chosen on a sender-disjoint calibration slice and reported on the "
                  "evaluation slice; no threshold was tuned on the reported rows.")
    interp.append("Single seed / single fold: no statistical significance is claimed.")

    txt = "\n".join([
        "=== PIAD-V2X threshold calibration (evaluation only) ===",
        f"data={data_path}  rows={manifest['rows']}  seed={seed}  fold={manifest['fold']}  "
        f"holdout={list(holdout)}  cal_frac={args.cal_frac}  device={device}",
        "",
        "## Observed (measured; threshold chosen on calibration slice, metrics on evaluation slice)",
        "",
        "AUC by model (evaluation slice):",
        render(["Model", "AUC"], auc_rows),
        "",
        "Operating points:",
        render(headers, rows),
        "",
        "## Interpretation",
        *("- " + s for s in interp),
    ])
    print("\n" + txt)

    (ckpt / "calibration.json").write_text(json.dumps({
        "data": str(data_path), "rows": manifest["rows"], "seed": seed,
        "fold": manifest["fold"], "holdout": list(holdout), "cal_frac": args.cal_frac,
        "device": str(device), "n_calibration": int(cal.sum()), "n_evaluation": int(ev.sum()),
        "target_fpr": args.fpr, "results": results, "interpretation": interp,
    }, indent=2))
    with (ckpt / "calibration.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
        w.writerow([])
        w.writerow(["Model", "AUC(eval)"])
        w.writerows(auc_rows)
    (ckpt / "calibration.txt").write_text(txt + "\n")
    print(f"\nwrote calibration.json / calibration.csv / calibration.txt to {ckpt}/")


if __name__ == "__main__":
    main()
