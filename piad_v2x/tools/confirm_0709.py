"""H1n confirmatory: does the multi-vehicle-scale AUC lift survive in denser _0709
rush-hour traffic? Within-detector scale ablation (kinematic-scale-only vs full
multi-scale), sender group-split - not a rival-detector contest."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from piad_v2x.models.relational import relational_features, KINEMATIC, REL_TRUE
rng=np.random.default_rng(42)
for scen in ["DataReplay_0709","DataReplaySybil_0709"]:
    df=pd.read_parquet(f"data/features/{scen}.parquet"); df=df[df.attackerType>=0].reset_index(drop=True)
    df=df.merge(relational_features(df),on="messageID",how="left")
    y=df["is_attacker"].values
    ug=np.unique(df.sender.values); rng.shuffle(ug); te=set(ug[:int(0.3*len(ug))])
    m=np.array([g in te for g in df.sender.values])
    def rf(cols):
        c=RandomForestClassifier(n_estimators=120,max_depth=16,n_jobs=-1,class_weight="balanced",random_state=42)
        c.fit(df.loc[~m,cols].fillna(0).values,y[~m])
        return round(roc_auc_score(y[m],c.predict_proba(df.loc[m,cols].fillna(0).values)[:,1]),3)
    sv=rf(KINEMATIC); pr=rf(KINEMATIC+REL_TRUE); rel=rf(REL_TRUE)
    atk_nn=df[y==1]["I1_occ_min_dist"].median(); ben_nn=df[y==0]["I1_occ_min_dist"].median()
    print(f"{scen:22s} full-SV={sv} full+REL={pr} lift={round(pr-sv,3)} rel-only={rel} "
          f"| occ_min_dist median attacker={atk_nn:.1f} benign={ben_nn:.1f}", flush=True)
