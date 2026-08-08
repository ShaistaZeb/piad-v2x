#!/usr/bin/env python3
"""
DECISIVE multi-vehicle-scale ablation.

Question: how much of the hard-class detection does the multi-vehicle interaction scale
carry? We ablate it from one multi-scale physics detector (kinematic-scale-only vs full)
and measure the cost. Invariants are computed as DIRECT physical quantities (no trained
model):

  I1 mutual occupancy exclusion : distance to nearest other vehicle (same 1s
     snapshot); small => physical overlap.
  I2 reachability / continuity  : (a) own-step displacement / (v_max*dt) ratio
     (teleport); (b) distance to nearest vehicle present one snapshot earlier
     (appearance without a physical predecessor).
  I3 collision feasibility (TTC): min time-to-collision to approaching neighbours;
     small => unresolved imminent collision that genuine traffic avoids.

This is a within-detector SCALE ABLATION, not a rival-detector contest: the
kinematic-scale-only variant (single-vehicle motion residuals r_pos_cv, r_spd_pos already
in the features) vs the full multi-scale detector (kinematic + multi-vehicle interaction
invariants). We report per-class univariate ROC-AUC (attacker vs benign) for each
observable, and a leakage-safe (sender group-split) RF on each. It localises which scale
carries the coordinated-attack signal: on the hard classes, full multi-scale AUC >>
single-vehicle AUC supports the coupling claim; parity refutes it.
"""
import glob, os, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

VMAX = 50.0    # m/s upper bound
DMIN = 5.0     # m vehicle footprint
R = 150.0      # m neighbour radius for TTC
HARD = {"DataReplay_1416","DataReplaySybil_1416","Disruptive_1416","EventualStop_1416"}
# FULL single-vehicle set (data-only variant) - fair baseline.
# NB: disp & dt are here, so the "own-step reachability" ratio is derivable -> it
# belongs to single-vehicle, NOT to the relational set.
KINEMATIC = ["px","py","sx","sy","ax","ay","hx","hy","spd_mag","acl_mag",
             "dt","disp","implied_spd","has_prev","r_pos_cv","r_spd_pos"]
# TRULY relational invariants: each REQUIRES other vehicles to compute.
REL_TRUE = ["I1_occ_min_dist","I2_reach_prev","I3_min_ttc","I3_n_collision"]
REL = ["I1_occ_min_dist","I2_reach_own","I2_reach_prev","I3_min_ttc","I3_n_collision"]  # for univariate report only


def relational_features(df):
    em = df.drop_duplicates("messageID")[["messageID","senderPseudo","sendTime",
                                          "px","py","sx","sy","disp","dt"]].reset_index(drop=True)
    em["tbin"] = em["sendTime"].round().astype(int)
    N=len(em)
    pos=em[["px","py"]].values; vel=em[["sx","sy"]].values
    occ=np.full(N,1e4); ttc=np.full(N,1e4); ncol=np.zeros(N); reachp=np.full(N,1e4)
    groups=sorted(em.groupby("tbin").groups.items())   # time order
    prev_pts=None; prev_bin=None
    for b,idx in groups:
        rows=np.asarray(idx)                # positional row indices (reset index)
        P=pos[rows]; V=vel[rows]
        if len(rows)>=2:
            tree=cKDTree(P)
            dd,_=tree.query(P,k=2); occ[rows]=dd[:,1]     # nearest OTHER vehicle
            for a in range(len(rows)):
                nbr=tree.query_ball_point(P[a],R); best=1e4; cc=0
                for j in nbr:
                    if j==a: continue
                    dp=P[j]-P[a]; dv=V[j]-V[a]; d=np.hypot(dp[0],dp[1])
                    if d<1e-6: best=0.0; cc+=1; continue
                    closing=-(dp[0]*dv[0]+dp[1]*dv[1])/d
                    if closing>0.1:
                        t=d/closing
                        if t<best: best=t
                        if t<3.0: cc+=1
                ttc[rows[a]]=best; ncol[rows[a]]=cc
        if prev_pts is not None and len(prev_pts)>0 and (b-prev_bin)<=3:
            reachp[rows]=cKDTree(prev_pts).query(P,k=1)[0]
        prev_pts=P; prev_bin=b
    em["I1_occ_min_dist"]=occ
    em["I3_min_ttc"]=ttc
    em["I3_n_collision"]=ncol
    em["I2_reach_prev"]=reachp
    em["I2_reach_own"]=(em["disp"]/(VMAX*em["dt"].clip(lower=0.1))).fillna(0).values
    return em[["messageID"]+REL]


def auc_oriented(y, s):
    s=np.nan_to_num(s, nan=np.nanmedian(s))
    if len(np.unique(y))<2: return float("nan")
    a=roc_auc_score(y,s)
    return max(a,1-a)   # separation regardless of sign


def main():
    scen_paths = sorted(glob.glob("data/features/*.parquet"))
    rng=np.random.default_rng(42)
    report={}
    for p in scen_paths:
        scen=os.path.basename(p).replace(".parquet","")
        df=pd.read_parquet(p); df=df[df.attackerType>=0].reset_index(drop=True)
        rel=relational_features(df)
        df=df.merge(rel, on="messageID", how="left")   # attach per-messageID
        y=df["is_attacker"].values
        # univariate separation (single-vehicle residuals + all candidate invariants)
        uni={f: round(auc_oriented(y, df[f].values),3) for f in ["r_pos_cv","r_spd_pos"]+REL}
        # scale ablation, group-split RF: kinematic-scale-only vs full (kinematic + multi-vehicle invariants)
        gid=df["sender"].values
        ug=np.unique(gid); rng.shuffle(ug)
        te=set(ug[:int(0.3*len(ug))]); mask_te=np.array([g in te for g in gid])
        def rf_auc(cols):
            Xtr=df.loc[~mask_te,cols].fillna(0).values; ytr=y[~mask_te]
            Xte=df.loc[mask_te,cols].fillna(0).values; yte=y[mask_te]
            if len(np.unique(ytr))<2 or len(np.unique(yte))<2: return float("nan")
            m=RandomForestClassifier(n_estimators=120,max_depth=16,n_jobs=-1,
                                     class_weight="balanced",random_state=42)
            m.fit(Xtr,ytr); return round(roc_auc_score(yte,m.predict_proba(Xte)[:,1]),3)
        full_sv=rf_auc(KINEMATIC); full_plus=rf_auc(KINEMATIC+REL_TRUE); rel_only=rf_auc(REL_TRUE)
        report[scen]={"univariate":uni,
                      "rf_full_single_vehicle":full_sv,
                      "rf_full_plus_relational":full_plus,
                      "rf_relational_only":rel_only,
                      "lift":round((full_plus or 0)-(full_sv or 0),3),
                      "hard":scen in HARD}
        r=report[scen]; tag="HARD" if r["hard"] else "easy"
        print(f"[{scen:22s} {tag}] kin-only={full_sv} multiscale={full_plus} lift={r['lift']} "
              f"rel-only={rel_only} | uni(true-rel) I1={uni['I1_occ_min_dist']} "
              f"I2prev={uni['I2_reach_prev']} I3ttc={uni['I3_min_ttc']} "
              f"| I2own(SV)={uni['I2_reach_own']}", flush=True)
    os.makedirs("experiments/results",exist_ok=True)
    json.dump(report, open("experiments/results/invariant_experiment.json","w"), indent=2)
    # hard-class summary
    print("\n=== HARD-CLASS SUMMARY (scale ablation: kinematic-only vs full multi-scale, RF AUC) ===")
    for s,r in report.items():
        if r["hard"]:
            print(f"  {s:22s} full-SV={r['rf_full_single_vehicle']}  full+REL={r['rf_full_plus_relational']}  "
                  f"lift={r['lift']}  (rel-only={r['rf_relational_only']})")
    print("wrote experiments/results/invariant_experiment.json")

if __name__=="__main__":
    main()
