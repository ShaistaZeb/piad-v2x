"""Evaluate trained PIAD-V2X models on the held-out unseen-attack split.

Inference only. This script never trains, never modifies the checkpoints, and
treats manifest.json as the single source of truth: it rebuilds the exact
leakage-safe test split train.py used (same data / rows / seed / fold /
holdout) and reloads each saved model, then reports honest binary
benign-vs-misbehaviour metrics plus per-held-out-attack recall.

Usage:
    python piad_v2x/tools/inference.py --ckpt checkpoints/
    python piad_v2x/tools/inference.py --ckpt checkpoints/ --data /abs/path/veremi.parquet  # override data location

Writes evaluation.json / evaluation.csv / evaluation.txt into the checkpoint
directory and prints the same summary. Works unchanged for smoke-test and full
200k checkpoints because every split parameter comes from the manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Make `import piad_v2x` work whether or not the package is pip-installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import joblib
import torch

from piad_v2x.data_utils.dataset import DatasetSpec, load_messages, split_holdout_attacks
from piad_v2x.models.detector import DetectorConfig, PINNDetector, evaluate_detector, predict_proba
from piad_v2x.models.eval_holdout import binary_report, per_class_report
from piad_v2x.data_utils.feature_extractor import extract_features

# Manifest model key -> display name, in report order.
MODEL_ORDER = [("rf", "Random Forest"), ("pinn_kin", "PINN-kin"), ("pinn_full", "PINN-full")]


def die(msg: str) -> None:
    sys.exit(f"inference.py: {msg}")


def fmt(x: float) -> str:
    return f"{x:.4f}" if x == x else "nan"   # x != x is True only for NaN


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def rf_attack_proba(rf, X: np.ndarray) -> np.ndarray:
    """P(any misbehaviour) = 1 - P(benign class 0)."""
    if 0 in rf.classes_:
        zero_col = list(rf.classes_).index(0)
        return 1.0 - rf.predict_proba(X)[:, zero_col]
    return rf.predict_proba(X).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PIAD-V2X models (inference only).")
    ap.add_argument("--model_dir", "--ckpt", dest="ckpt", type=Path, default=Path("checkpoints"),
                    help="trained-model directory containing manifest.json + models")
    ap.add_argument("--data", type=Path, default=None,
                    help="override only the CSV path (all split params still come from the manifest)")
    args = ap.parse_args()

    ckpt = args.ckpt
    man_path = ckpt / "manifest.json"
    if not man_path.exists():
        die(f"manifest.json not found in {ckpt}/ . Run piad_v2x/tools/train.py first.")
    manifest = json.loads(man_path.read_text())

    # --- resolve dataset (manifest is the source of truth for everything else) ---
    data_path = Path(args.data) if args.data else Path(manifest["data"])
    if not data_path.exists():
        die(f"dataset not found: {data_path}\n"
            f"Run from the repo root, or pass --data with the correct CSV path.")

    seed = manifest["seed"]
    holdout = tuple(int(c) for c in manifest["holdout"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- rebuild the identical leakage-safe test split -------------------------
    spec = DatasetSpec(
        csv_path=str(data_path),
        n_rows=None if manifest["rows"] == 0 else manifest["rows"],
        fold_index=manifest["fold"],
        random_state=seed,
    )
    df = load_messages(spec)
    _, test_df = split_holdout_attacks(df, holdout, spec)
    feats = extract_features(test_df)
    X, y = feats.X, feats.y
    print(f"reconstructed test split: {len(test_df):,} messages "
          f"({int((y != 0).sum()):,} attack / {int((y == 0).sum()):,} benign), holdout={holdout}")

    # --- reload each model and run inference -----------------------------------
    metrics: dict[str, dict] = {}
    per_attack: dict[str, tuple] = {}
    for key, name in MODEL_ORDER:
        meta = manifest["models"].get(key)
        if meta is None:
            die(f"manifest has no entry for model '{key}'.")
        mfile = ckpt / meta["file"]
        if not mfile.exists():
            die(f"missing model file: {mfile} (referenced by manifest['models']['{key}']).")

        if meta["type"] == "RandomForest":
            rf = joblib.load(mfile)
            pred = rf.predict(X)
            p_attack = rf_attack_proba(rf, X)
        elif meta["type"] == "PINN":
            cfg = DetectorConfig(**meta["config"])
            model = PINNDetector(cfg).to(device)
            model.load_state_dict(torch.load(mfile, map_location=device))
            model.eval()
            pred = evaluate_detector(model, X, y, device=device)
            p_attack = 1.0 - predict_proba(model, X, device=device)[:, 0]
        else:
            die(f"unknown model type '{meta['type']}' for '{key}'.")

        rep = binary_report(y, pred, p_attack)
        pcr = per_class_report(y, pred, holdout)
        metrics[name] = {
            "auc": rep.auc_roc, "f1": rep.f1_attack, "precision": rep.precision_attack,
            "recall": rep.recall_attack, "accuracy": rep.accuracy,
            "n_attack": rep.n_attack, "n_benign": rep.n_benign,
        }
        per_attack[name] = pcr.recall

    # --- comparison table ------------------------------------------------------
    main_headers = ["Model", "AUC", "F1", "Precision", "Recall", "Accuracy"]
    main_rows = [[name, fmt(m["auc"]), fmt(m["f1"]), fmt(m["precision"]),
                  fmt(m["recall"]), fmt(m["accuracy"])]
                 for name, m in metrics.items()]

    # --- per-held-out-attack recall table --------------------------------------
    pa_headers = ["Held-out attack"] + [name for _, name in MODEL_ORDER]
    pa_rows = []
    for i, c in enumerate(holdout):
        pa_rows.append([str(c)] + [fmt(per_attack[name][i]) for _, name in MODEL_ORDER])

    # --- honest verdict (no spin): PINN-full vs RandomForest -------------------
    rf_m, full_m = metrics["Random Forest"], metrics["PINN-full"]
    verdict_lines = []
    for metric in ("auc", "f1"):
        rfv, fv = rf_m[metric], full_m[metric]
        if rfv != rfv or fv != fv:
            verdict_lines.append(f"{metric.upper()}: not comparable (nan).")
        elif fv > rfv:
            verdict_lines.append(f"{metric.upper()}: PINN-full ({fv:.4f}) > Random Forest ({rfv:.4f}).")
        elif fv < rfv:
            verdict_lines.append(f"{metric.upper()}: Random Forest ({rfv:.4f}) > PINN-full ({fv:.4f}); "
                                 f"PINN-full does NOT beat the baseline on this split.")
        else:
            verdict_lines.append(f"{metric.upper()}: PINN-full ties Random Forest ({fv:.4f}).")

    # --- assemble text report --------------------------------------------------
    txt = "\n".join([
        "=== PIAD-V2X evaluation (held-out unseen-attack split) ===",
        f"data={data_path}  rows={manifest['rows']}  seed={seed}  fold={manifest['fold']}  "
        f"holdout={list(holdout)}  device={device}",
        "",
        "Binary detection (benign vs misbehaviour):",
        render_table(main_headers, main_rows),
        "",
        "Per-held-out-attack recall (fraction of each unseen family flagged as misbehaviour):",
        render_table(pa_headers, pa_rows),
        "",
        "Verdict:",
        *("  " + v for v in verdict_lines),
    ])
    print("\n" + txt)

    # --- persist (new files only; existing checkpoints untouched) --------------
    (ckpt / "evaluation.json").write_text(json.dumps({
        "data": str(data_path), "rows": manifest["rows"], "seed": seed,
        "fold": manifest["fold"], "holdout": list(holdout), "device": str(device),
        "n_test": int(len(test_df)),
        "metrics": metrics,
        "per_attack_recall": {"classes": list(holdout),
                              **{name: list(per_attack[name]) for _, name in MODEL_ORDER}},
        "verdict": verdict_lines,
    }, indent=2))

    with (ckpt / "evaluation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(main_headers)
        w.writerows(main_rows)
        w.writerow([])
        w.writerow(pa_headers)
        w.writerows(pa_rows)

    (ckpt / "evaluation.txt").write_text(txt + "\n")
    print(f"\nwrote evaluation.json / evaluation.csv / evaluation.txt to {ckpt}/")


if __name__ == "__main__":
    main()
