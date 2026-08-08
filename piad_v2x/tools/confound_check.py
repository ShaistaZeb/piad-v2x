"""Is the relational lift genuine physics or a density/location confound?
Tests, for the two win classes: (1) does local DENSITY alone explain it?
(2) do the relational invariants still lift AUC when density is added to the
single-vehicle baseline? (3) which relational features carry the weight?"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from piad_v2x.models.relational import relational_features, KINEMATIC, REL_TRUE, auc_oriented

def density(df):
    em=df.drop_duplicates("messageID")[["messageID","sendTime","px","py"]].reset_index(drop=True)
    em["tbin"]=em["sendTime"].round().astype(int); N=len(em); pos=em[["px","py"]].values
    dens=np.zeros(N)
    for b,idx in em.groupby("tbin").groups.items():
        rows=np.asarray(idx); P=pos[rows]
        if len(rows)>=2:
            t=cKDTree(P); dens[rows]=t.query_ball_point(P,100.0,return_length=True)-1
    em["nbr_cnt"]=dens; return em[["messageID","nbr_cnt"]]

for scen in ["DataReplay_1416","DataReplaySybil_1416"]:
    df=pd.read_parquet(f"data/features/{scen}.parquet"); df=df[df.attackerType>=0].reset_index(drop=True)
    df=df.merge(relational_features(df),on="messageID",how="left").merge(density(df),on="messageID",how="left")
    y=df["is_attacker"].values
    rng=np.random.default_rng(1); ug=np.unique(df.sender.values); rng.shuffle(ug)
    te=set(ug[:int(0.3*len(ug))]); m=np.array([g in te for g in df.sender.values])
    def rf(cols,imp=False):
        clf=RandomForestClassifier(n_estimators=120,max_depth=16,n_jobs=-1,class_weight="balanced",random_state=1)
        clf.fit(df.loc[~m,cols].fillna(0).values,y[~m])
        a=roc_auc_score(y[m],clf.predict_proba(df.loc[m,cols].fillna(0).values)[:,1])
        return (round(a,3),dict(zip(cols,np.round(clf.feature_importances_,3)))) if imp else round(a,3)
    dens_auc=round(auc_oriented(y,df["nbr_cnt"].values),3)
    atk_d=df[y==1].nbr_cnt.median(); ben_d=df[y==0].nbr_cnt.median()
    sv=rf(KINEMATIC); sv_dens=rf(KINEMATIC+["nbr_cnt"]); sv_dens_rel=rf(KINEMATIC+["nbr_cnt"]+REL_TRUE)
    rel_auc,imp=rf(REL_TRUE,imp=True)
    print(f"\n=== {scen} ===")
    print(f"density(nbr_cnt) univariate AUC={dens_auc}  | attacker median nbrs={atk_d:.1f} vs benign={ben_d:.1f}")
    print(f"full-SV={sv}  SV+density={sv_dens}  SV+density+RELATIONAL={sv_dens_rel}  "
          f"=> relational lift OVER SV+density = {round(sv_dens_rel-sv_dens,3)}")
    print(f"relational-only AUC={rel_auc}  feature importances={imp}")
