"""Build the loss-based detector mixed table from the per-scenario parquet.

This is the join that collapses the project to ONE data pipeline. the loss-based phase (the
physics-in-loss detector) originally consumed a separate pre-flattened MixAll CSV;
that CSV is retired. Both studies now derive from
the SAME raw extraction:

    raw VeReMi Extension zips
        -> piad_v2x/tools/veremi_extract.py        (per-scenario parquet, multi-scale detector)
        -> piad_v2x/tools/build_mixed_table.py    (this file: mixed table for the loss-based detector)

Run from the repository root:

    python piad_v2x/tools/build_mixed_table.py --out data/veremi.parquet

The column map is deterministic and verified by inspection against the
extractor in piad_v2x/tools/veremi_extract.py.

Two correctness points baked in:
  1. Column rename: the extractor px/py/sx/sy/ax/ay/hx/hy -> the loaders
     posx/posy/spdx/spdy/aclx/acly/hedx/hedy; attackerType -> class.
  2. ID NAMESPACING: each scenario is an independent simulation, so raw `sender` /
     `senderPseudo` ids COLLIDE across scenarios. We prefix them with the scenario name
     so split_by_sender() groups the right vehicle and no two scenarios' vehicles merge
     under one id - without this the leakage-safe split is silently wrong.
"""
import argparse, glob, os
import pandas as pd

# extractor column -> loss-based loader column (piad_v2x/dataset.py SIM_COLS)
RENAME = {
    "px": "posx", "py": "posy",
    "sx": "spdx", "sy": "spdy",
    "ax": "aclx", "ay": "acly",
    "hx": "hedx", "hy": "hedy",
}
# columns load_messages() selects (noise-delta cols are optional and absent
# from the extracted table; the loss-based detector does not use them).
STUDY1_COLS = ["type", "sendTime", "sender", "senderPseudo", "messageID", "class",
               "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy"]


def build_one(path):
    scen = os.path.basename(path).replace(".parquet", "")
    df = pd.read_parquet(path)
    df = df[df["attackerType"] >= 0].copy()          # drop unknown-sender rows
    df = df.rename(columns=RENAME)
    df["class"] = df["attackerType"].astype(int)      # 0 = benign, k = attack type
    df["type"] = 3                                    # all rows are received BSMs
    # namespace ids so vehicles/pseudonyms never collide across scenarios
    df["sender"] = scen + ":" + df["sender"].astype(str)
    df["senderPseudo"] = scen + ":" + df["senderPseudo"].astype(str)
    keep = [c for c in STUDY1_COLS if c in df.columns]
    missing = [c for c in STUDY1_COLS if c not in df.columns]
    if missing:
        print(f"  [{scen}] WARNING missing columns {missing} (skipped)", flush=True)
    return df[keep], scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features",
                    help="dir of per-scenario parquet from veremi_extract.py")
    ap.add_argument("--pattern", default="*.parquet",
                    help="glob within --features (e.g. '*_1416.parquet' for the "
                         "consistent sparse window; default = all scenarios)")
    ap.add_argument("--out", default="data/veremi.parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.features, args.pattern)))
    if not paths:
        raise SystemExit(f"No parquet in {args.features}. Run piad_v2x/tools/veremi_extract.py "
                         "first (see data/README.md).")
    frames = []
    for p in paths:
        f, scen = build_one(p)
        print(f"[{scen:24s}] {len(f):>9,} rows  classes={sorted(f['class'].unique())}",
              flush=True)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {len(out):,} rows across {len(frames)} scenarios -> {args.out}")
    print("Loss-based detector: point --data at this file "
          "(load_messages reads .parquet natively).")


if __name__ == "__main__":
    main()
