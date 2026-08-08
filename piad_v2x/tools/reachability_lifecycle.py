#!/usr/bin/env python3
"""
Step 4: reachability-gated pseudonym lifecycle (title's 2nd half).

Claim: a PHYSICAL reachability gate neutralises Sybil pseudonyms that the
aggregated-TRUST gate misses, because trust resets on each new pseudonym (measured:
~28% of Sybil pseudonyms ever neutralised) whereas reachability is a hard physical
check that does not reset. A Sybil replaying different vehicles' messages emits a
physically-discontinuous trajectory: consecutive claimed positions of ONE
pseudonym jump farther than v_max*dt.

Three lifecycle gates on the Sybil scenario, same accepted-malicious outcome
(malicious messages accepted before neutralisation) + benign false-revocation cost:
  C       per-message detector flag only (no lifecycle action)
  D_trust aggregated detector-trust EMA gate
  D_reach reachability gate: revoke a pseudonym at its first physical
          discontinuity (|dpos| > v_max*dt)
"""
import json, numpy as np, pandas as pd, joblib
from piad_v2x.models.relational import relational_features

VMAX=50.0; ALPHA=0.8; TAU=0.15; K=3; NMIN=5; SCEN="DataReplaySybil_0709"  # TAU=0.15 = committed operating point (aligned 2026-07-18)
MODELS="experiments/results/models"

def main():
    df=pd.read_parquet(f"data/features/{SCEN}.parquet"); df=df[df.attackerType>=0].reset_index(drop=True)
    rf=joblib.load(f"{MODELS}/rf.joblib"); sc=joblib.load(f"{MODELS}/scalers.joblib")
    X=sc["sc_all"].transform(df[sc["KINEMATIC"]+sc["RESIDUALS"]].values)
    df["p_mal"]=rf.predict_proba(X)[:,1]; thr=sc["thr_rf"]
    df["flag"]=(df["p_mal"]>=thr).astype(int)

    acc={"C":0,"D_trust":0,"D_reach":0}; n_atk_msg=0
    atk_ps=set(); neu_trust=set(); neu_reach=set()
    ben_ps=set(); ben_falserevoke_trust=set(); ben_falserevoke_reach=set()

    for ps,g in df.sort_values("sendTime").groupby("senderPseudo"):
        g=g.reset_index(drop=True)
        is_atk = g["is_attacker"].max()==1
        (atk_ps if is_atk else ben_ps).add(ps)
        # trust EMA + reachability along the pseudonym's stream
        T=0.5; belowT=0; t_trust=np.inf; t_reach=np.inf
        px=py=pt=None
        for i,r in g.iterrows():
            # reachability: physical discontinuity within THIS pseudonym
            if px is not None:
                dt=r["sendTime"]-pt; dt=dt if dt>0 else 1.0
                jump=np.hypot(r["px"]-px, r["py"]-py)
                if jump > VMAX*dt and np.isinf(t_reach):
                    t_reach=r["sendTime"]
            px,py,pt=r["px"],r["py"],r["sendTime"]
            # trust EMA
            T=ALPHA*T+(1-ALPHA)*(1-r["p_mal"])
            if i>=NMIN-1:
                if T<TAU:
                    belowT+=1
                    if belowT>=K and np.isinf(t_trust): t_trust=r["sendTime"]
                else: belowT=0
        if is_atk:
            if np.isfinite(t_trust): neu_trust.add(ps)
            if np.isfinite(t_reach): neu_reach.add(ps)
            atk=g[g["is_attacker"]==1]
            n_atk_msg+=len(atk)
            acc["C"]      += int((atk["flag"]==0).sum())
            acc["D_trust"]+= int(((atk["flag"]==0)&(atk["sendTime"]<t_trust)).sum())
            acc["D_reach"]+= int(((atk["flag"]==0)&(atk["sendTime"]<t_reach)).sum())
        else:
            if np.isfinite(t_trust): ben_falserevoke_trust.add(ps)
            if np.isfinite(t_reach): ben_falserevoke_reach.add(ps)

    out={"scenario":SCEN,"n_attacker_msgs":int(n_atk_msg),
         "attacker_pseudonyms":len(atk_ps),"benign_pseudonyms":len(ben_ps),
         "neutralised_pct":{"D_trust":round(len(neu_trust)/max(1,len(atk_ps)),3),
                            "D_reach":round(len(neu_reach)/max(1,len(atk_ps)),3)},
         "accepted_malicious":{k:acc[k] for k in acc},
         "accepted_rate":{k:round(acc[k]/max(1,n_atk_msg),3) for k in acc},
         "benign_false_revocation_pct":{"D_trust":round(len(ben_falserevoke_trust)/max(1,len(ben_ps)),3),
                                        "D_reach":round(len(ben_falserevoke_reach)/max(1,len(ben_ps)),3)}}
    json.dump(out,open("experiments/results/reachability_lifecycle.json","w"),indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
