"""VeReMi suitability for scene-level interaction graphs: measures whether the
observables needed (co-temporal neighbours, relative motion, synchronised
timestamps, graph connectivity) actually exist, with real numbers."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

def analyse(path, label):
    df = pd.read_parquet(path)
    df = df[df.attackerType >= 0]
    # unique emitted messages (dedup receiver duplication)
    em = df.drop_duplicates("messageID")[["messageID","senderPseudo","sender","sendTime",
                                          "px","py","sx","sy","is_attacker"]].copy()
    em["tbin"] = em["sendTime"].round().astype(int)
    print(f"\n===== {label} =====")
    print(f"emitted messages: {len(em):,} | unique senders: {em.sender.nunique():,} "
          f"| time span: {em.sendTime.min():.0f}-{em.sendTime.max():.0f}s")
    # timestamp synchronisation: how many distinct vehicles share a 1s snapshot
    sizes = em.groupby("tbin").senderPseudo.nunique()
    print(f"1s snapshots: {sizes.shape[0]:,} | vehicles/snapshot mean={sizes.mean():.1f} "
          f"median={sizes.median():.0f} max={sizes.max()}")
    # neighbour / graph connectivity across a sample of snapshots
    rng = np.random.default_rng(0)
    bins = rng.choice(sizes.index.values, size=min(400, len(sizes)), replace=False)
    deg50=[]; deg100=[]; deg200=[]; mind=[]; frac_conn=[]
    atk_mind=[]; ben_mind=[]
    for b in bins:
        g = em[em.tbin==b]
        if len(g) < 2: continue
        pts = g[["px","py"]].values
        tree = cKDTree(pts)
        for r,acc in ((50,deg50),(100,deg100),(200,deg200)):
            counts = tree.query_ball_point(pts, r, return_length=True) - 1  # exclude self
            acc.extend(counts.tolist())
        # nearest-neighbour distance
        dd,_ = tree.query(pts, k=2)
        nn = dd[:,1]
        mind.extend(nn.tolist())
        frac_conn.append(float((tree.query_ball_point(pts,100,return_length=True)-1 >=1).mean()))
        isatk = g["is_attacker"].values.astype(bool)
        atk_mind.extend(nn[isatk].tolist()); ben_mind.extend(nn[~isatk].tolist())
    def q(a): a=np.array(a); return f"mean={a.mean():.1f} median={np.median(a):.1f} p90={np.percentile(a,90):.1f}"
    print(f"neighbours within  50m: {q(deg50)}")
    print(f"neighbours within 100m: {q(deg100)}")
    print(f"neighbours within 200m: {q(deg200)}")
    print(f"fraction of vehicles with >=1 neighbour @100m: {np.mean(frac_conn):.3f}")
    print(f"nearest-neighbour dist (m): {q(mind)}")
    print(f"  nearest-nbr dist BENIGN:  median={np.median(ben_mind):.1f}m")
    print(f"  nearest-nbr dist ATTACKER:median={np.median(atk_mind):.1f}m  "
          f"(overlap<5m: benign={np.mean(np.array(ben_mind)<5):.3f} attacker={np.mean(np.array(atk_mind)<5):.3f})")
    # relative motion availability
    nonzero_v = (em[["sx","sy"]].abs().sum(axis=1) > 0).mean()
    print(f"velocity vectors present & non-zero: {nonzero_v:.3f}  -> relative motion computable")
    # who-heard-whom communication graph
    recv = df.groupby("receiver").sender.nunique()
    print(f"comm graph: distinct senders heard per receiver mean={recv.mean():.1f} median={recv.median():.0f}")

for p,l in [("data/features/DataReplaySybil_1416.parquet","DataReplaySybil (hard, Sybil)"),
            ("data/features/DataReplay_1416.parquet","DataReplay (hard)"),
            ("data/features/ConstPos_1416.parquet","ConstPos (easy)")]:
    analyse(p,l)
