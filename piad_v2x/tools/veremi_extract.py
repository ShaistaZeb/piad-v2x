#!/usr/bin/env python3
"""
VeReMi Extension feature extractor (PIAD-V2X).

Streams the real raw dataset (nested zips of per-receiver JSON logs) and emits a
compact per-message feature table. Designed for a 16 GB / 54 GB-free machine:
never bulk-extracts to disk; reads inner zips into memory one at a time.

Message model (confirmed from the raw files):
  type 2 -> the receiver's own GPS fix (ego), no sender field.
  type 3 -> a received BSM with sendTime, sender (TRUE vehicle id, oracle),
            senderPseudo (pseudonym the sender used), messageID, and the
            claimed pos/spd/acl/hed (+ per-axis noise).
  type 4 -> (ground-truth file) the TRUE transmitted state per messageID.

Labelling: each trace filename encodes the *vehicle's* attacker type as A<k>
(A0 = genuine, A1.. = attacker). We build vehId -> attackerType from filenames
and label a received message by its (true) sender's attacker type. This is the
standard VeReMi sender-level ground truth.

LEAKAGE BOUNDARY (enforced downstream, documented here): `sender` and the label
are ORACLE columns. The detector may use only the claimed kinematics and
senderPseudo-linked history. `sender` is used solely for labels and for
group-splitting (all messages from one vehicle stay in one fold).

Physics residuals are computed per (receiver, senderPseudo) sequence from the
constant-velocity / constant-acceleration motion model. They are emitted as
columns so the ablation can (ii) feed them as features and (iii) use them as the
in-loss physics target, while (i) the data-only arm ignores them.
"""
import argparse
import io
import json
import math
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

# Raw VeReMi Extension scenario zips. NOT bundled (see data/README.md for the
# download link). Default is the in-repo data/raw/; override with --raw-dir or the
# PIAD_VEREMI_RAW environment variable so nothing points outside a fresh clone.
DATASET_DIR = os.environ.get("PIAD_VEREMI_RAW", "data/raw")


def parse_attacker_type(fname):
    """traceJSON-<vehId>-<x>-A<k>-<t>-<m>.json -> (vehId:int, attackerType:int) or None."""
    base = os.path.basename(fname)
    if not base.startswith("traceJSON-"):
        return None
    parts = base.split("-")
    # parts: ['traceJSON', vehId, x, 'A<k>', t, 'm.json']
    try:
        veh_id = int(parts[1])
        atk_tok = parts[3]
        if not atk_tok.startswith("A"):
            return None
        atk = int(atk_tok[1:])
        return veh_id, atk
    except (IndexError, ValueError):
        return None


def norm2(x, y):
    return math.sqrt(x * x + y * y)


def build_ground_truth(lines):
    """Parse a traceGroundTruthJSON log -> {messageID: true-state dict}.

    Type-4 records log the TRUE transmitted state of each message (before any
    attacker falsification), keyed by messageID. Joining this onto the type-3
    RECEIVED state lets a goal-preserving adaptive attacker be defined: the attack
    magnitude is |reported - true|, which the attacker must preserve while it tries
    to minimise the detector's residual score. Benign messages have reported ~= true
    (sensor noise only); attacker messages have reported = falsified, true = real.
    """
    gt = {}
    for ln in lines:
        if not ln:
            continue
        try:
            m = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if m.get("type") != 4:
            continue
        mid = m.get("messageID", -1)
        pos = m.get("pos", [0.0, 0.0, 0.0])
        spd = m.get("spd", [0.0, 0.0, 0.0])
        acl = m.get("acl", [0.0, 0.0, 0.0])
        hed = m.get("hed", [0.0, 0.0, 0.0])
        gt[mid] = {
            "true_px": float(pos[0]), "true_py": float(pos[1]),
            "true_sx": float(spd[0]), "true_sy": float(spd[1]),
            "true_ax": float(acl[0]), "true_ay": float(acl[1]),
            "true_hx": float(hed[0]), "true_hy": float(hed[1]),
            "true_sender": m.get("sender", -1),
        }
    return gt


_TRUE_COLS = ("true_px", "true_py", "true_sx", "true_sy",
              "true_ax", "true_ay", "true_hx", "true_hy")


def features_for_receiver(lines, atk_map, receiver_veh, scenario, window, gt_map=None):
    """Yield per-received-message feature dicts for one receiver's log."""
    gt_map = gt_map or {}
    # collect type-3 messages grouped by senderPseudo, keep order by rcvTime
    by_pseudo = defaultdict(list)
    for ln in lines:
        if not ln:
            continue
        try:
            m = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if m.get("type") != 3:
            continue
        by_pseudo[m.get("senderPseudo")].append(m)

    for pseudo, msgs in by_pseudo.items():
        msgs.sort(key=lambda d: d.get("rcvTime", 0.0))
        prev = None
        for m in msgs:
            pos = m.get("pos", [0.0, 0.0, 0.0])
            spd = m.get("spd", [0.0, 0.0, 0.0])
            acl = m.get("acl", [0.0, 0.0, 0.0])
            hed = m.get("hed", [0.0, 0.0, 0.0])
            px, py = float(pos[0]), float(pos[1])
            sx, sy = float(spd[0]), float(spd[1])
            ax, ay = float(acl[0]), float(acl[1])
            hx, hy = float(hed[0]), float(hed[1])
            spd_mag = norm2(sx, sy)
            acl_mag = norm2(ax, ay)
            sender = m.get("sender", -1)
            atk = atk_map.get(sender, -1)

            row = {
                "scenario": scenario,
                "window": window,
                "receiver": receiver_veh,
                "senderPseudo": pseudo,
                "sender": sender,           # ORACLE (label/group only)
                "messageID": m.get("messageID", -1),
                "rcvTime": float(m.get("rcvTime", 0.0)),
                "sendTime": float(m.get("sendTime", 0.0)),
                "px": px, "py": py, "sx": sx, "sy": sy,
                "ax": ax, "ay": ay, "hx": hx, "hy": hy,
                "spd_mag": spd_mag, "acl_mag": acl_mag,
                "attackerType": atk,
                "is_attacker": 1 if atk > 0 else 0,
            }

            # join true transmitted state (type-4) by messageID; has_truth=0 if absent
            truth = gt_map.get(m.get("messageID", -1))
            if truth is None:
                row.update({c: float("nan") for c in _TRUE_COLS})
                row["has_truth"] = 0
            else:
                for c in _TRUE_COLS:
                    row[c] = truth[c]
                row["has_truth"] = 1

            if prev is None:
                row.update(dict(has_prev=0, dt=0.0, disp=0.0,
                                r_pos_cv=0.0, r_pos_ca=0.0, r_spd=0.0,
                                r_spd_pos=0.0, r_hed=0.0, implied_spd=0.0))
            else:
                dt = row["rcvTime"] - prev["rcvTime"]
                if dt <= 0:
                    dt = 1.0
                # constant-velocity position prediction from previous claim
                pred_px = prev["px"] + prev["sx"] * dt
                pred_py = prev["py"] + prev["sy"] * dt
                r_pos_cv = norm2(px - pred_px, py - pred_py)
                # constant-acceleration position prediction
                pred_px_ca = prev["px"] + prev["sx"] * dt + 0.5 * prev["ax"] * dt * dt
                pred_py_ca = prev["py"] + prev["sy"] * dt + 0.5 * prev["ay"] * dt * dt
                r_pos_ca = norm2(px - pred_px_ca, py - pred_py_ca)
                # speed prediction from previous speed + accel
                pred_sx = prev["sx"] + prev["ax"] * dt
                pred_sy = prev["sy"] + prev["ay"] * dt
                r_spd = norm2(sx - pred_sx, sy - pred_sy)
                # implied speed from displacement vs claimed speed
                disp_x, disp_y = px - prev["px"], py - prev["py"]
                disp = norm2(disp_x, disp_y)
                impl_sx, impl_sy = disp_x / dt, disp_y / dt
                r_spd_pos = norm2(sx - impl_sx, sy - impl_sy)
                implied_spd = norm2(impl_sx, impl_sy)
                # heading vs velocity-vector consistency (1 - cos), only if moving
                if spd_mag > 0.5 and norm2(hx, hy) > 1e-6:
                    cos = (sx * hx + sy * hy) / (spd_mag * norm2(hx, hy))
                    r_hed = 1.0 - max(-1.0, min(1.0, cos))
                else:
                    r_hed = 0.0
                row.update(dict(has_prev=1, dt=dt, disp=disp,
                                r_pos_cv=r_pos_cv, r_pos_ca=r_pos_ca, r_spd=r_spd,
                                r_spd_pos=r_spd_pos, r_hed=r_hed, implied_spd=implied_spd))
            prev = dict(px=px, py=py, sx=sx, sy=sy, ax=ax, ay=ay,
                        rcvTime=row["rcvTime"])
            yield row


def process_inner_zip(inner_bytes, scenario, window, max_receivers=None):
    zf = zipfile.ZipFile(io.BytesIO(inner_bytes))
    names = zf.namelist()
    trace_names = [n for n in names if os.path.basename(n).startswith("traceJSON-")]
    gt_names = [n for n in names
                if os.path.basename(n).startswith("traceGroundTruthJSON")]
    # vehId -> attackerType map from ALL trace filenames in this window
    atk_map = {}
    for n in trace_names:
        pa = parse_attacker_type(n)
        if pa is not None:
            atk_map[pa[0]] = pa[1]
    # messageID -> true transmitted state, merged across this window's GT logs
    gt_map = {}
    for n in gt_names:
        try:
            gt_lines = zf.read(n).decode("utf-8", errors="ignore").splitlines()
        except (KeyError, zipfile.BadZipFile):
            continue
        gt_map.update(build_ground_truth(gt_lines))
    print(f"    [{window}] ground-truth messages: {len(gt_map)}", flush=True)
    if max_receivers is not None:
        trace_names = trace_names[:max_receivers]

    rows = []
    for i, n in enumerate(trace_names):
        pa = parse_attacker_type(n)
        receiver_veh = pa[0] if pa else -1
        try:
            data = zf.read(n).decode("utf-8", errors="ignore")
        except (KeyError, zipfile.BadZipFile):
            continue
        lines = data.splitlines()
        rows.extend(features_for_receiver(lines, atk_map, receiver_veh, scenario, window, gt_map))
        if (i + 1) % 200 == 0:
            print(f"    [{window}] {i+1}/{len(trace_names)} receivers, {len(rows)} rows", flush=True)
    return rows, atk_map


def process_scenario(scenario, out_path, max_receivers=None, raw_dir=None):
    zip_path = os.path.join(raw_dir or DATASET_DIR, scenario + ".zip")
    if not os.path.exists(zip_path):
        sys.exit(f"scenario zip not found: {zip_path}")
    print(f"[{scenario}] opening outer zip", flush=True)
    all_rows = []
    with zipfile.ZipFile(zip_path) as outer:
        inner_names = [n for n in outer.namelist() if n.lower().endswith(".zip")]
        for inner_name in inner_names:
            # window tag from inner filename: VeReMi_<start>_<end>_...
            base = os.path.basename(inner_name)
            window = "_".join(base.split("_")[1:3]) if base.startswith("VeReMi_") else base
            print(f"  [{scenario}] reading inner window {window} ({inner_name})", flush=True)
            inner_bytes = outer.read(inner_name)
            rows, atk_map = process_inner_zip(inner_bytes, scenario, window, max_receivers)
            n_atk_veh = sum(1 for v in atk_map.values() if v > 0)
            print(f"  [{scenario}/{window}] vehicles={len(atk_map)} attacker_vehicles={n_atk_veh} "
                  f"messages={len(rows)}", flush=True)
            all_rows.extend(rows)
            del inner_bytes

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{scenario}] wrote {len(df)} rows -> {out_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-receivers", type=int, default=None,
                    help="cap receivers per window (validation runs)")
    ap.add_argument("--raw-dir", default=None,
                    help="directory of raw scenario zips; defaults to PIAD_VEREMI_RAW "
                         "env var or data/raw (see data/README.md)")
    args = ap.parse_args()
    df = process_scenario(args.scenario, args.out, args.max_receivers, args.raw_dir)

    # quick sanity summary
    print("\n=== SANITY ===", flush=True)
    print("rows:", len(df))
    print("class balance (is_attacker):")
    print(df["is_attacker"].value_counts(normalize=True).round(3).to_dict())
    print("attackerType counts:", df["attackerType"].value_counts().to_dict())
    hp = df[df["has_prev"] == 1]
    if len(hp):
        print("median r_pos_cv  benign vs attacker:",
              round(hp[hp.is_attacker == 0]["r_pos_cv"].median(), 3), "vs",
              round(hp[hp.is_attacker == 1]["r_pos_cv"].median(), 3))
        print("median r_spd_pos benign vs attacker:",
              round(hp[hp.is_attacker == 0]["r_spd_pos"].median(), 3), "vs",
              round(hp[hp.is_attacker == 1]["r_spd_pos"].median(), 3))


if __name__ == "__main__":
    main()
