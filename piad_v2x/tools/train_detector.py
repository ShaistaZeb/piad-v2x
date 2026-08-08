#!/usr/bin/env python3
"""
Detector placement study on real VeReMi Extension features.

Three matched-capacity neural arms + two baselines, group-split by true sender,
evaluated exactly as pre-registered:

  Variant i    data-only MLP        (kinematics only, BCE)
  Variant ii   residual-as-feature  (kinematics + physics residuals, BCE)
  Variant iii  residual-in-loss PINN(kinematics only inputs; class head + state head;
                                 loss = BCE + lambda * motion-model residual)
  RF       Random Forest on Variant-ii features (strong classical baseline; compute cost)
  RuleThr  non-learned residual-threshold detector (pure physics reference)

Outputs: experiments/results/detector_metrics.json (+ per-arm test predictions
saved for the cluster-bootstrap step in stats_bootstrap.py). NO test-set peeking:
threshold + lambda + RF depth chosen on validation only.
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, brier_score_loss)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
FEAT_DIR = "data/features"
OUT_DIR = "experiments/results"

KINEMATIC = ["px","py","sx","sy","ax","ay","hx","hy","spd_mag","acl_mag",
             "dt","disp","implied_spd","has_prev"]
RESIDUALS = ["r_pos_cv","r_pos_ca","r_spd","r_spd_pos","r_hed"]
HARD = {"DataReplay_1416","DataReplaySybil_1416","Disruptive_1416","EventualStop_1416"}

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_all():
    frames = []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet"))):
        df = pd.read_parquet(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["attackerType"] >= 0].copy()          # drop unresolved senders
    df["attack_class"] = np.where(df["is_attacker"] == 1, df["scenario"], "benign")
    return df


def sender_split(df):
    """60/20/20 split over unique BARE sender groups, stratified by attacker.
    Bare sender (not scenario#sender): the same physical vehicle follows a near-
    identical benign trajectory across scenarios, so scenario#sender would let one
    vehicle's benign motion appear in both train and test. Grouping by bare sender
    keeps all of a vehicle's rows (every scenario) in one fold."""
    rng = np.random.default_rng(SEED)
    df["gid"] = df["sender"].astype(str)
    g = df.groupby("gid")["is_attacker"].max().reset_index()   # group label
    parts = {}
    for lab in (0, 1):
        gids = np.array(g[g.is_attacker == lab]["gid"], dtype=object)
        rng.shuffle(gids)
        n = len(gids); a, b = int(0.6*n), int(0.8*n)
        parts.setdefault("train", []).extend(gids[:a])
        parts.setdefault("val", []).extend(gids[a:b])
        parts.setdefault("test", []).extend(gids[b:])
    assign = {gid: k for k, lst in parts.items() for gid in lst}
    df["split"] = df["gid"].map(assign)
    return df


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), pinn=False):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]; d = h
        self.trunk = nn.Sequential(*layers)
        self.cls = nn.Linear(d, 1)
        self.pinn = pinn
        self.state = nn.Linear(d, 4) if pinn else None   # predicts px,py,sx,sy

    def forward(self, x):
        z = self.trunk(x)
        logit = self.cls(z).squeeze(-1)
        state = self.state(z) if self.pinn else None
        return logit, state


def train_torch(Xtr, ytr, Xva, yva, n_in, pinn=False, lam=0.0,
                phys_ctx_tr=None, phys_ctx_va=None, phys_mu=None, phys_sd=None,
                epochs=40, bs=4096):
    """phys_ctx rows: (px,py,sx,sy,ax,ay,dt) RAW; the in-loss physics term compares
    the state head's output to the motion-model extrapolation of the CURRENT claim,
    both in STANDARDIZED state space (phys_mu/phys_sd for px,py,sx,sy) so the term
    is O(1) and does not swamp the BCE."""
    model = MLP(n_in, pinn=pinn).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_w = torch.tensor([float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum()))],
                         dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEV)
    if pinn:
        Ptr = torch.tensor(phys_ctx_tr, dtype=torch.float32, device=DEV)
        mu = torch.tensor(phys_mu, dtype=torch.float32, device=DEV)
        sd = torch.tensor(phys_sd, dtype=torch.float32, device=DEV)
    n = len(Xtr_t); best_va = 1e9; best_state = None; patience = 6; bad = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            logit, state = model(Xtr_t[idx])
            loss = bce(logit, ytr_t[idx])
            if pinn and lam > 0:
                p = Ptr[idx]  # px,py,sx,sy,ax,ay,dt (RAW)
                dt = p[:, 6]
                # motion-model extrapolation of the current claimed state
                tgt = torch.stack([p[:, 0] + p[:, 2]*dt, p[:, 1] + p[:, 3]*dt,
                                   p[:, 2] + p[:, 4]*dt, p[:, 3] + p[:, 5]*dt], dim=1)
                tgt = (tgt - mu) / sd            # standardized target, O(1)
                phys = ((state - tgt) ** 2).mean()
                loss = loss + lam * phys
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogit, _ = model(Xva_t)
            vloss = bce(vlogit, torch.tensor(yva, dtype=torch.float32, device=DEV)).item()
        if vloss < best_va - 1e-4:
            best_va = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        logit, _ = model(torch.tensor(X, dtype=torch.float32, device=DEV))
        return torch.sigmoid(logit).cpu().numpy()


def pick_threshold(yva, pva, target_recall=0.90):
    """Highest threshold achieving >= target recall on validation (vectorised)."""
    total_pos = float((yva == 1).sum())
    if total_pos == 0:
        return 0.5
    order = np.argsort(-pva)
    ys = yva[order]; ps = pva[order]
    recall = np.cumsum(ys) / total_pos          # nondecreasing as threshold drops
    idx = int(np.searchsorted(recall, target_recall))
    idx = min(idx, len(ps) - 1)
    return float(ps[idx])


def per_class_f1(df_test, yprob, thr, hard_only=True):
    yhat = (yprob >= thr).astype(int)
    benign_mask = df_test["is_attacker"].values == 0
    out = {}
    classes = df_test["attack_class"].unique()
    for c in classes:
        if c == "benign":
            continue
        m = benign_mask | (df_test["attack_class"].values == c)
        f1 = f1_score(df_test["is_attacker"].values[m], yhat[m], zero_division=0)
        rec = recall_score(df_test["is_attacker"].values[m], yhat[m], zero_division=0)
        out[c] = {"f1": round(float(f1), 4), "recall": round(float(rec), 4),
                  "n_attack": int((df_test["attack_class"].values == c).sum())}
    hard = [c for c in out if c in HARD]
    macro_hard = float(np.mean([out[c]["f1"] for c in hard])) if hard else None
    return out, macro_hard


def eval_block(name, df_test, yprob, thr):
    y = df_test["is_attacker"].values
    yhat = (yprob >= thr).astype(int)
    perclass, macro_hard = per_class_f1(df_test, yprob, thr)
    is_prob = float(np.nanmin(yprob)) >= 0.0 and float(np.nanmax(yprob)) <= 1.0
    return {
        "variant": name,
        "roc_auc": round(float(roc_auc_score(y, yprob)), 4),
        "pr_auc": round(float(average_precision_score(y, yprob)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "fpr": round(float(((yhat == 1) & (y == 0)).sum() / max(1, (y == 0).sum())), 4),
        "brier": round(float(brier_score_loss(y, yprob)), 4) if is_prob else None,
        "threshold": round(float(thr), 4),
        "macro_f1_hard": None if macro_hard is None else round(macro_hard, 4),
        "per_class": perclass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sample", type=int, default=0, help="optional row cap for a fast smoke run")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_all()
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=SEED).reset_index(drop=True)
    df = sender_split(df)
    print("rows:", len(df), "| split:", df["split"].value_counts().to_dict(),
          "| attacker frac:", round(df.is_attacker.mean(), 3), "| device:", DEV, flush=True)
    # export sender -> split so the closed loop / adaptive attacker can restrict
    # scoring to held-out (test) senders and avoid train-on-test contamination.
    split_map = df.drop_duplicates("sender").set_index("sender")["split"].to_dict()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "split_map.json"), "w") as f:
        json.dump({str(k): v for k, v in split_map.items()}, f)
    print("split senders:", {s: sum(1 for v in split_map.values() if v == s)
                             for s in ("train", "val", "test")}, flush=True)

    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))

    # feature matrices
    sc_kin = StandardScaler().fit(tr[KINEMATIC].values)
    sc_all = StandardScaler().fit(tr[KINEMATIC + RESIDUALS].values)
    Xk = {k: sc_kin.transform(d[KINEMATIC].values).astype(np.float32) for k, d in (("tr",tr),("va",va),("te",te))}
    Xa = {k: sc_all.transform(d[KINEMATIC+RESIDUALS].values).astype(np.float32) for k,d in (("tr",tr),("va",va),("te",te))}
    y = {k: d["is_attacker"].values.astype(float) for k,d in (("tr",tr),("va",va),("te",te))}

    # physics context (RAW, unscaled) for the in-loss term: prev state + dt
    def phys_ctx(d):
        # reconstruct prev state from current claim and lag columns is not exact;
        # instead use current claim as the anchor and dt: the in-loss term
        # regularises the predicted NEXT state toward the motion model applied to
        # the CURRENT claimed state (observable at inference).
        return np.stack([d["px"].values, d["py"].values, d["sx"].values, d["sy"].values,
                         d["ax"].values, d["ay"].values, d["dt"].values.clip(min=0.1)], axis=1)
    P = {k: phys_ctx(d) for k,d in (("tr",tr),("va",va),("te",te))}

    results = {}

    # ---- Variant i: data-only ----
    print("training Variant i (data-only)...", flush=True)
    m_i = train_torch(Xk["tr"], y["tr"], Xk["va"], y["va"], Xk["tr"].shape[1],
                      pinn=False, epochs=args.epochs)
    p_i_va = predict_torch(m_i, Xk["va"]); thr_i = pick_threshold(y["va"], p_i_va)
    p_i_te = predict_torch(m_i, Xk["te"])
    results["variant_i_data_only"] = eval_block("variant_i_data_only", te, p_i_te, thr_i)
    results["variant_i_data_only"]["n_params"] = sum(p.numel() for p in m_i.parameters())

    # ---- Variant ii: residual-as-feature ----
    print("training Variant ii (residual-as-feature)...", flush=True)
    m_ii = train_torch(Xa["tr"], y["tr"], Xa["va"], y["va"], Xa["tr"].shape[1],
                       pinn=False, epochs=args.epochs)
    p_ii_va = predict_torch(m_ii, Xa["va"]); thr_ii = pick_threshold(y["va"], p_ii_va)
    p_ii_te = predict_torch(m_ii, Xa["te"])
    results["variant_ii_residual_feature"] = eval_block("variant_ii_residual_feature", te, p_ii_te, thr_ii)
    results["variant_ii_residual_feature"]["n_params"] = sum(p.numel() for p in m_ii.parameters())

    # ---- Variant iii: residual-in-loss PINN (lambda selected on val) ----
    # standardization stats for the physics target (px,py,sx,sy = KINEMATIC[0:4])
    phys_mu = sc_kin.mean_[0:4]; phys_sd = sc_kin.scale_[0:4]
    best = None
    for lam in (0.01, 0.1, 1.0):
        print(f"training Variant iii PINN lambda={lam}...", flush=True)
        m = train_torch(Xk["tr"], y["tr"], Xk["va"], y["va"], Xk["tr"].shape[1],
                        pinn=True, lam=lam, phys_ctx_tr=P["tr"], phys_ctx_va=P["va"],
                        phys_mu=phys_mu, phys_sd=phys_sd, epochs=args.epochs)
        pva = predict_torch(m, Xk["va"])
        score = average_precision_score(y["va"], pva)
        if best is None or score > best[0]:
            best = (score, lam, m)
    _, lam_star, m_iii = best
    p_iii_va = predict_torch(m_iii, Xk["va"]); thr_iii = pick_threshold(y["va"], p_iii_va)
    p_iii_te = predict_torch(m_iii, Xk["te"])
    results["variant_iii_residual_in_loss"] = eval_block("variant_iii_residual_in_loss", te, p_iii_te, thr_iii)
    results["variant_iii_residual_in_loss"]["n_params"] = sum(p.numel() for p in m_iii.parameters())
    results["variant_iii_residual_in_loss"]["lambda_star"] = lam_star

    # ---- RF baseline (on Variant-ii features) ----
    # RF trains on a stratified subsample to bound memory on 16 GB (neural arms
    # use all data). A documented deviation for memory bounds.
    print("training RF baseline...", flush=True)
    rf_cap = 1_500_000
    if len(Xa["tr"]) > rf_cap:
        rng2 = np.random.default_rng(SEED)
        idx = rng2.choice(len(Xa["tr"]), size=rf_cap, replace=False)
        Xrf, yrf = Xa["tr"][idx], y["tr"][idx]
    else:
        Xrf, yrf = Xa["tr"], y["tr"]
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=120, max_depth=22, n_jobs=-1,
                                class_weight="balanced", random_state=SEED)
    rf.fit(Xrf, yrf)
    rf_fit_s = time.time() - t0
    p_rf_va = rf.predict_proba(Xa["va"])[:, 1]; thr_rf = pick_threshold(y["va"], p_rf_va)
    p_rf_te = rf.predict_proba(Xa["te"])[:, 1]
    results["rf_baseline"] = eval_block("rf_baseline", te, p_rf_te, thr_rf)
    results["rf_baseline"]["fit_seconds"] = round(rf_fit_s, 1)

    # ---- Rule threshold (pure physics) ----
    print("evaluating rule-threshold detector...", flush=True)
    res_score_va = va[RESIDUALS].max(axis=1).values
    res_score_te = te[RESIDUALS].max(axis=1).values
    # calibrate threshold on val to hit recall 0.90
    thr_rule = pick_threshold(y["va"], res_score_va)
    results["rule_threshold"] = eval_block("rule_threshold", te,
                                           res_score_te, thr_rule)

    # ---- compute axis: inference latency (batch 1 AND 1024) + peak mem ----
    def latency_torch(model, X, bs, reps):
        model.eval(); Xt = torch.tensor(X[:bs], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            for _ in range(3): model(Xt)  # warmup
            t0 = time.time()
            for _ in range(reps): model(Xt)
            return (time.time()-t0)/reps/bs*1000  # ms per message
    lat_iii_1 = latency_torch(m_iii, Xk["te"], 1, 200)
    lat_iii_1024 = latency_torch(m_iii, Xk["te"], 1024, 20)
    t0 = time.time()
    for _ in range(200): rf.predict_proba(Xa["te"][:1])
    lat_rf_1 = (time.time()-t0)/200*1000
    t0 = time.time(); rf.predict_proba(Xa["te"][:1024]); lat_rf_1024 = (time.time()-t0)/1024*1000
    import pickle
    rf_mem_mb = len(pickle.dumps(rf)) / 1e6
    pinn_mem_mb = sum(p.numel()*4 for p in m_iii.parameters()) / 1e6
    results["compute_cost"] = {
        "variant_iii_ms_per_msg_batch1": round(lat_iii_1, 5),
        "variant_iii_ms_per_msg_batch1024": round(lat_iii_1024, 5),
        "rf_ms_per_msg_batch1": round(lat_rf_1, 5),
        "rf_ms_per_msg_batch1024": round(lat_rf_1024, 5),
        "variant_iii_model_MB": round(pinn_mem_mb, 3),
        "rf_model_MB": round(rf_mem_mb, 1),
        "device": DEV}

    with open(os.path.join(OUT_DIR, "detector_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # persist models + scalers for the closed loop (D-5) and adaptive attacker (D-6)
    mdir = os.path.join(OUT_DIR, "models"); os.makedirs(mdir, exist_ok=True)
    joblib.dump(rf, os.path.join(mdir, "rf.joblib"))
    joblib.dump({"sc_kin": sc_kin, "sc_all": sc_all,
                 "KINEMATIC": KINEMATIC, "RESIDUALS": RESIDUALS,
                 "thr_rf": float(thr_rf), "thr_iii": float(thr_iii)},
                os.path.join(mdir, "scalers.joblib"))
    torch.save({"state_dict": {k: v.cpu() for k, v in m_iii.state_dict().items()},
                "n_in": Xk["tr"].shape[1], "lambda_star": lam_star},
               os.path.join(mdir, "variant_iii.pt"))

    # also persist test predictions for the cluster bootstrap
    np.savez(os.path.join(OUT_DIR, "test_preds.npz"),
             y=y["te"], gid=te["gid"].values, attack_class=te["attack_class"].values,
             variant_i=p_i_te, variant_ii=p_ii_te, variant_iii=p_iii_te, rf=p_rf_te, rule=res_score_te,
             thr_i=thr_i, thr_ii=thr_ii, thr_iii=thr_iii, thr_rf=thr_rf, thr_rule=thr_rule)

    print("\n=== SUMMARY (test) ===")
    for k in ("variant_i_data_only","variant_ii_residual_feature","variant_iii_residual_in_loss",
              "rf_baseline","rule_threshold"):
        r = results[k]
        print(f"{k:28s} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
              f"F1={r['f1']:.3f} macroF1_hard={r['macro_f1_hard']} FPR={r['fpr']:.3f}")
    print("lambda*:", results["variant_iii_residual_in_loss"].get("lambda_star"))
    print("compute:", results["compute_cost"])
    print("wrote", os.path.join(OUT_DIR, "detector_metrics.json"))


if __name__ == "__main__":
    main()
