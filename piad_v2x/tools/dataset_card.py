#!/usr/bin/env python3
"""Build experiments/dataset_card.md: real N per scenario (rows, senders, class
balance, residual separation). Data description only - not a hypothesis test."""
import glob, os
import numpy as np, pandas as pd

FEAT_DIR = "data/features"; OUT = "experiments/dataset_card.md"
os.makedirs("experiments", exist_ok=True)
rows = []
tot_msg = tot_snd = 0
for p in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet"))):
    d = pd.read_parquet(p); d = d[d.attackerType >= 0]
    scen = os.path.basename(p).replace(".parquet", "")
    n = len(d)
    snd = d.groupby("sender").is_attacker.max()
    n_snd = snd.shape[0]; n_atk_snd = int((snd == 1).sum())
    atk_frac = round(float(d.is_attacker.mean()), 3)
    hp = d[d.has_prev == 1]
    rb = round(float(hp[hp.is_attacker == 0].r_pos_cv.median()), 2) if len(hp) else float("nan")
    ra = round(float(hp[hp.is_attacker == 1].r_pos_cv.median()), 2) if len(hp) else float("nan")
    rows.append((scen, n, n_snd, n_atk_snd, atk_frac, rb, ra))
    tot_msg += n; tot_snd += n_snd

lines = ["# Dataset Card - VeReMi Extension features (PIAD-V2X, relational phase)\n",
         "Real raw VeReMi Extension (CC-BY 4.0), low-density `_1416` windows, "
         "features from `piad_v2x/tools/veremi_extract.py`. Rows = received BSMs after "
         "dropping unresolved senders (attackerType = -1).\n",
         "| Scenario | Messages | Senders | Attacker senders | Attacker msg frac | med r_pos_cv benign | med r_pos_cv attacker |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]} | {r[5]} | {r[6]} |")
lines.append(f"\n**Totals:** {tot_msg:,} messages across {tot_snd:,} sender-instances "
             f"in {len(rows)} scenarios.\n")
lines.append("Sender-instances are the group unit for the leakage-safe split "
             "(60/20/20). The effective N for sender-clustered inference is the "
             "sender count, not the message count.")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nwrote", OUT)
