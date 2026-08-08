"""Train the three PIAD-V2X detector variants and save them for evaluation.

Trains, on the leakage-safe unseen-attack split (kinematic-falsification
families held out entirely from training):

  * RF baseline   kinematic-only RandomForest        -> rf.joblib
  * PINN kin-only physics prior L_kin only            -> pinn_kin.pt
  * PINN full     multi-physics L_kin + L_lwr         -> pinn_full.pt

A manifest.json records every parameter needed to rebuild the identical split
and reload the models, so piad_v2x/tools/inference.py reconstructs the exact test set.

Usage:
    python piad_v2x/tools/train.py --data data/veremi.parquet --rows 200000 --epochs 6 --out checkpoints/
    python piad_v2x/tools/train.py --data data/veremi.parquet --rows 0          # full dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Make `import piad_v2x` work whether or not the package is pip-installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import torch

from piad_v2x.models.baseline_kin import BaselineConfig, train_baseline
from piad_v2x.data_utils.dataset import (
    DatasetSpec,
    class_distribution,
    load_messages,
    split_holdout_attacks,
)
from piad_v2x.models.detector import DetectorConfig, PINNDetector, train_detector
from piad_v2x.data_utils.feature_extractor import FEATURE_NAMES, extract_features


def main() -> None:
    p = argparse.ArgumentParser(description="Train PIAD-V2X detector variants.")
    p.add_argument("--data", type=Path, default=Path("data/veremi.parquet"),
                   help="path to the VeReMi CSV")
    p.add_argument("--rows", type=int, default=200_000,
                   help="stratified subsample size; 0 = full dataset")
    p.add_argument("--epochs", type=int, default=6, help="PINN training epochs")
    p.add_argument("--fold", type=int, default=0, help="StratifiedGroupKFold test fold")
    p.add_argument("--seed", type=int, default=17, help="random seed")
    p.add_argument("--holdout", type=int, nargs="+", default=[1, 3, 5, 7, 9],
                   help="attack classes held out from training (unseen-attack protocol)")
    p.add_argument("--out", type=Path, default=Path("checkpoints"),
                   help="output directory for models + manifest")
    p.add_argument("--amp", action="store_true",
                   help="enable CUDA automatic mixed precision for the neural "
                        "detector variants (inert on CPU and for the random forest)")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    if not args.data.exists():
        sys.exit(f"dataset not found: {args.data}\nSee README.md (Data) for the download.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout = tuple(int(c) for c in args.holdout)
    log(f"=== PIAD-V2X training ===")
    log(f"  data={args.data}  rows={args.rows}  epochs={args.epochs}")
    log(f"  fold={args.fold}  seed={args.seed}  holdout={holdout}  device={device}")

    # --- data + leakage-safe unseen-attack split --------------------------
    spec = DatasetSpec(
        csv_path=str(args.data),
        n_rows=None if args.rows == 0 else args.rows,
        fold_index=args.fold,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    df = load_messages(spec)
    log(f"\nloaded {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    train_df, test_df = split_holdout_attacks(df, holdout, spec)
    overlap = len(set(train_df["sender"]) & set(test_df["sender"]))
    log(f"split: train={len(train_df):,}  test={len(test_df):,}  sender_overlap={overlap}")
    log("train class distribution (held-out classes must be absent):")
    log(class_distribution(train_df).to_string())

    feats = extract_features(train_df)
    send_time = train_df["sendTime"].to_numpy(dtype="float64")
    log(f"features: X={feats.X.shape}  ({len(FEATURE_NAMES)} per message)")

    models_meta: dict[str, dict] = {}

    # --- 1. RF baseline ---------------------------------------------------
    log("\n[1/3] RandomForest baseline ...")
    t0 = time.perf_counter()
    rf_cfg = BaselineConfig(random_state=args.seed)
    rf = train_baseline(feats.X, feats.y, rf_cfg)
    joblib.dump(rf, out / "rf.joblib")
    log(f"  saved rf.joblib  ({time.perf_counter() - t0:.1f}s)")
    models_meta["rf"] = {"file": "rf.joblib", "type": "RandomForest",
                         "config": asdict(rf_cfg)}

    # --- 2 + 3. PINN kin-only and full multi-physics ----------------------
    for name, lam_lwr in (("pinn_kin", 0.0), ("pinn_full", 0.5)):
        idx = 2 if name == "pinn_kin" else 3
        log(f"\n[{idx}/3] PINN {name} (lambda_kin=0.5, lambda_lwr={lam_lwr}) ...")
        t0 = time.perf_counter()
        cfg = DetectorConfig(epochs=args.epochs, lambda_kin=0.5,
                             lambda_lwr=lam_lwr, seed=args.seed)
        model, res = train_detector(feats.X, feats.y, send_time, cfg=cfg, device=device,
                                    use_amp=args.amp)
        torch.save(model.state_dict(), out / f"{name}.pt")
        log(f"  final loss={res.final_loss:.4f}  saved {name}.pt  "
            f"({time.perf_counter() - t0:.1f}s)")
        models_meta[name] = {"file": f"{name}.pt", "type": "PINN",
                             "config": asdict(cfg)}

    # --- manifest ---------------------------------------------------------
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": str(args.data),
        "rows": args.rows,
        "fold": args.fold,
        "seed": args.seed,
        "holdout": list(holdout),
        "n_classes": DetectorConfig().n_classes,
        "feature_names": list(FEATURE_NAMES),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "sender_overlap": overlap,
        "models": models_meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "train_log.txt").write_text("\n".join(log_lines))
    log(f"\nwrote manifest.json + 3 models to {out}/")


if __name__ == "__main__":
    main()
