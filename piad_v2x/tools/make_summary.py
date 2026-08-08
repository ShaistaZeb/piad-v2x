"""Assemble checkpoints/experiment_summary.md from the measured artifacts.

Reads manifest.json + evaluation.json (the single sources of truth produced by
train.py / inference.py) and emits a dissertation-ready summary that strictly
separates the Observed result (measured numbers, verbatim) from Interpretation
(ordering + caveats). It invents no numbers and makes no statistical claim:
a single seed / single fold cannot establish significance, and the summary
says so explicitly.

Usage:
    python piad_v2x/tools/make_summary.py --ckpt checkpoints/ \
        --train-seconds 214 --eval-seconds 33
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

MODELS = ["Random Forest", "PINN-kin", "PINN-full"]


def f(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.4f}" if x == x else "nan"


def human_size(n: int) -> str:
    return f"{n/1e6:.1f} MB" if n >= 1e6 else f"{n/1e3:.1f} KB"


def hms(sec) -> str:
    if sec is None:
        return "not recorded"
    sec = int(sec)
    return f"{sec//60}m {sec%60}s ({sec}s)" if sec >= 60 else f"{sec}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    ap.add_argument("--train-seconds", type=int, default=None)
    ap.add_argument("--eval-seconds", type=int, default=None)
    args = ap.parse_args()

    ckpt = args.ckpt
    manifest = json.loads((ckpt / "manifest.json").read_text())
    ev = json.loads((ckpt / "evaluation.json").read_text())
    metrics = ev["metrics"]
    pa = ev["per_attack_recall"]
    holdout = manifest["holdout"]
    epochs = manifest["models"]["pinn_full"]["config"]["epochs"]

    sizes = {m["file"]: (ckpt / m["file"]).stat().st_size
             for m in manifest["models"].values() if (ckpt / m["file"]).exists()}

    # ----- ordering (factual, from the measured numbers) -----
    def order_by(metric: str) -> list[tuple[str, float]]:
        vals = [(m, metrics[m][metric]) for m in MODELS]
        vals = [(m, v) for m, v in vals if isinstance(v, (int, float)) and v == v]
        return sorted(vals, key=lambda kv: kv[1], reverse=True)

    auc_rank = order_by("auc")
    f1_rank = order_by("f1")
    auc_lead = auc_rank[0][0] if auc_rank else "n/a"
    f1_lead = f1_rank[0][0] if f1_rank else "n/a"

    # degenerate "flag everything" detector: high recall, low accuracy
    degenerate = [m for m in MODELS
                  if metrics[m]["recall"] >= 0.99 and metrics[m]["accuracy"] < 0.5]

    L: list[str] = []
    L.append("# PIAD-V2X Experiment Summary")
    L.append("")
    L.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  |  "
             f"trained: {manifest.get('created', 'n/a')}  |  device: {ev.get('device', 'n/a')}")
    L.append("")
    L.append("## 1. Experiment configuration")
    L.append("")
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    L.append(f"| Dataset | `{manifest['data']}` |")
    L.append(f"| Subsample rows | {manifest['rows']:,} ({'full dataset' if manifest['rows'] == 0 else 'stratified'}) |")
    L.append(f"| Training messages | {manifest['n_train']:,} |")
    L.append(f"| Test messages | {ev['n_test']:,} |")
    L.append(f"| Held-out attack classes | {holdout} (removed entirely from training) |")
    L.append(f"| Sender overlap (train ∩ test) | {manifest['sender_overlap']} |")
    L.append(f"| Seed | {manifest['seed']} |  ")
    L.append(f"| Fold | {manifest['fold']} |")
    L.append(f"| PINN epochs | {epochs} |")
    L.append(f"| Features per message | {len(manifest['feature_names'])} |")
    L.append("")
    L.append("## 2. Runtime and artifacts")
    L.append("")
    L.append("| Item | Value |")
    L.append("|---|---|")
    L.append(f"| Training runtime | {hms(args.train_seconds)} |")
    L.append(f"| Evaluation runtime | {hms(args.eval_seconds)} |")
    for fname, sz in sizes.items():
        L.append(f"| `{fname}` | {human_size(sz)} |")
    L.append("")
    L.append("## 3. Observed result")
    L.append("")
    L.append("> Measured values, reported verbatim from `evaluation.json`. No interpretation in this section.")
    L.append("")
    L.append("### 3.1 Binary detection (benign vs misbehaviour)")
    L.append("")
    L.append("| Model | AUC | F1 | Precision | Recall | Accuracy |")
    L.append("|---|---|---|---|---|---|")
    for m in MODELS:
        d = metrics[m]
        L.append(f"| {m} | {f(d['auc'])} | {f(d['f1'])} | {f(d['precision'])} | "
                 f"{f(d['recall'])} | {f(d['accuracy'])} |")
    L.append("")
    L.append("### 3.2 Per-held-out-attack recall")
    L.append("")
    L.append("| Held-out attack | " + " | ".join(MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    for i, c in enumerate(holdout):
        L.append(f"| {c} | " + " | ".join(f(pa[m][i]) for m in MODELS) + " |")
    L.append("")
    L.append("## 4. Interpretation")
    L.append("")
    L.append(f"- **Highest AUC:** {auc_lead}. Ranking: "
             + " > ".join(f"{m} ({v:.4f})" for m, v in auc_rank) + ".")
    L.append(f"- **Highest F1:** {f1_lead}. Ranking: "
             + " > ".join(f"{m} ({v:.4f})" for m, v in f1_rank) + ".")
    if degenerate:
        L.append(f"- **Degenerate-detector warning:** {', '.join(degenerate)} show recall ≥ 0.99 with "
                 f"accuracy < 0.50, i.e. they flag almost all traffic as misbehaviour. Their high "
                 f"recall is not informative without the accompanying precision/accuracy.")
    L.append("- **No statistical claim.** This is a single run at one seed and one fold. The differences "
             "above are point estimates with no confidence intervals and no significance test. Establishing "
             "that any model is *statistically* better requires repeated seeds and/or cross-fold evaluation, "
             "which this experiment did not perform.")
    L.append("")
    L.append("## 5. Final verdict")
    L.append("")
    for v in ev.get("verdict", []):
        L.append(f"- {v}")
    L.append("- Statistical significance: **not established** (single seed/fold; see section 4).")
    L.append("")

    (ckpt / "experiment_summary.md").write_text("\n".join(L))
    print(f"wrote {ckpt / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
