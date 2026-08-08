#!/usr/bin/env python3
"""VeReMi NextGen (InTAS) extraction adapter -> the committed feature-table schema.

Schema verified by inspection 2026-07-19 (recorded in
the external-validity plan), NOT assumed:
  - sender{} block is the TRANSMITTED / CLAIMED state (detector input); confirmed on
    zeroSpeedReport (attacker reports spd 0 while its positions move ~13.8 m/s).
  - `attacker` (0/1) is the ground-truth label; `sender_alias`=pseudonym; `sender_id`=true id.
  - time is nanoseconds (divide by 1e9); positions are UTM metres; beacon rate 1 Hz.
  - speed is scalar m/s, heading is degrees in NAVIGATION convention (0=North, clockwise):
    verified vx=spd*sin(hed), vy=spd*cos(hed) matches motion direction to 2.5 deg median.
  - messageID is per-send (shared across receivers), so dedup by messageID = unique sends.
  - the separate ground-truth JSON is a distinct clean simulation and does NOT join; the core
    replication needs no true-position join (it scores the claimed state, labels by `attacker`).

Train and Test are DIFFERENT physical areas of InTAS, so absolute UTM position is not
transferable. Default pos_mode='center' subtracts each split's own coordinate origin (a pure
translation): it makes px,py ranges comparable across areas and leaves the relational invariants
(I1/I2/I3, computed from position differences) unchanged. pos_mode='raw' keeps UTM; the detector
feature-selection choice (whether to use absolute px,py at all) is left to the run stage.

Output columns match relational.KINEMATIC + REL_TRUE plus keys (messageID, senderPseudo, sender,
rcvTime, sendTime, is_attacker, attackerType), so the committed detector/lifecycle scripts consume
it unchanged.
"""
import argparse, io, json, os, zipfile, math
import numpy as np
import pandas as pd
from piad_v2x.models.relational import relational_features

ATTACK_CODE = {  # traceability only; is_attacker drives everything
    "benign": 0, "constantPositionOffset": 1, "randomPositionOffset": 2, "positionMirroring": 3,
    "constantSpeedOffset": 4, "randomSpeedOffset": 5, "zeroSpeedReport": 6, "suddenConstantSpeed": 7,
    "reversedHeading": 8, "feignedBraking": 9, "accelerationMultiplication": 10, "timeDelayAttack": 11,
    "dosAttack": 12, "trafficCongestionSybil": 13, "suddenStop": 14, "dataReplay": 15,
}


def parse_pos(p):
    return [float(x) for x in (p.split(",") if isinstance(p, str) else p)]


def load_split(zip_path, split):
    """Aggregate all per-receiver JSON message logs in one Train/Validation/Test nested zip."""
    zf = zipfile.ZipFile(zip_path)
    nz = [n for n in zf.namelist() if f"/{split}/" in n and n.endswith(".zip")]
    if not nz:
        raise SystemExit(f"no {split} nested zip found in {zip_path}")
    inner = zipfile.ZipFile(io.BytesIO(zf.read(nz[0])))
    rows = []
    for j in inner.namelist():
        if j.endswith(".json"):
            rows += json.loads(inner.read(j))
    return rows


def to_rows(msgs, atk_code):
    """Dedup received messages to unique sends (by messageID) in the committed schema."""
    seen = {}
    for m in msgs:
        mid = int(m["messageID"])
        if mid in seen:
            continue
        s = m["sender"]
        pos = parse_pos(s["pos"])
        spd = float(s["spd"]); acl = float(s["acl"]); h = math.radians(float(s["hed"]))
        sx, sy = spd * math.sin(h), spd * math.cos(h)          # nav convention (verified)
        hx, hy = math.sin(h), math.cos(h)
        ax, ay = acl * math.sin(h), acl * math.cos(h)          # accel along heading
        att = int(m["attacker"])
        seen[mid] = dict(
            messageID=mid, senderPseudo=int(m["sender_alias"]), sender=str(m["sender_id"]),
            rcvTime=int(m["rcvTime"]) / 1e9, sendTime=int(m["sendTime"]) / 1e9,
            is_attacker=att, attackerType=(atk_code if att == 1 else 0),
            px=pos[0], py=pos[1], sx=sx, sy=sy, ax=ax, ay=ay, hx=hx, hy=hy,
            spd_mag=spd, acl_mag=abs(acl),
        )
    return pd.DataFrame(seen.values())


def add_derived(df, pos_mode):
    if pos_mode == "center":
        df["px"] = df["px"] - df["px"].min()
        df["py"] = df["py"] - df["py"].min()
    df = df.sort_values(["senderPseudo", "sendTime"]).reset_index(drop=True)
    g = df.groupby("senderPseudo", sort=False)
    ppx, ppy = g["px"].shift(), g["py"].shift()
    psx, psy = g["sx"].shift(), g["sy"].shift()
    pt = g["sendTime"].shift()
    hp = pt.notna()
    dt = (df["sendTime"] - pt).where(hp, 0.0)
    dts = dt.where(dt > 0, 1.0)
    pred_px, pred_py = ppx + psx * dt, ppy + psy * dt
    r_pos_cv = np.hypot(df["px"] - pred_px, df["py"] - pred_py)
    dispx, dispy = df["px"] - ppx, df["py"] - ppy
    disp = np.hypot(dispx, dispy)
    impl_sx, impl_sy = dispx / dts, dispy / dts
    r_spd_pos = np.hypot(df["sx"] - impl_sx, df["sy"] - impl_sy)
    implied = np.hypot(impl_sx, impl_sy)
    df["has_prev"] = hp.astype(int)
    df["dt"] = dt.astype(float)
    df["disp"] = np.where(hp, disp, 0.0)
    df["implied_spd"] = np.where(hp, implied, 0.0)
    df["r_pos_cv"] = np.where(hp, r_pos_cv, 0.0)
    df["r_spd_pos"] = np.where(hp, r_spd_pos, 0.0)
    rel = relational_features(df)                              # I1/I2/I3, committed code verbatim
    df = df.merge(rel, on="messageID", how="left")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to an InTAS_<scenario>_<attack>.zip")
    ap.add_argument("--splits", nargs="*", default=["Train", "Validation", "Test"])
    ap.add_argument("--pos-mode", choices=["center", "raw"], default="center")
    ap.add_argument("--outdir", default="data/features_nextgen")
    args = ap.parse_args()
    stem = os.path.basename(args.zip)[:-4]                     # InTAS_highway_2_dataReplay
    attack = stem.split("_")[-1]
    atk_code = ATTACK_CODE.get(attack, 99)
    os.makedirs(args.outdir, exist_ok=True)
    for split in args.splits:
        msgs = load_split(args.zip, split)
        df = to_rows(msgs, atk_code)
        df = add_derived(df, args.pos_mode)
        out = os.path.join(args.outdir, f"{stem}_{split}.parquet")
        df.to_parquet(out)
        print(f"[{split:11s}] {len(df):6d} sends | attackers {int(df.is_attacker.sum()):5d} "
              f"({df.is_attacker.mean()*100:.1f}%) | -> {out}", flush=True)


if __name__ == "__main__":
    main()
