"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/run_surrogates.py

Usage: python scripts/run_surrogates.py outputs/swmm_results.csv
Trains ANN ensemble, GPR, PCE on log1p(flood volume); reports regression and flood/no-flood
metrics on validation, test and TAIL sets, and the speed-up against SWMM.
Writes outputs/surrogate_metrics.csv and outputs/surrogate_predictions_test.csv
"""
import sys, time, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m5_dataset import assemble as A
from culvert_twin.m6_surrogates.surrogates import ANNEnsemble, GPR, PCE

csv = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "swmm_results.csv")
reg = load_registry()
d, thr = A.split(A.load(csv, reg))
(ROOT / "outputs").mkdir(exist_ok=True)
tr = d[d.part == "train"]
X, y = tr[A.FEATURES].values, tr.y_vol.values
models = {"ANN_ensemble": ANNEnsemble(), "GPR_Matern52_ARD": GPR(), "PCE_deg3": PCE()}
sets = {"val": d[d.part == "val"], "test": d[d.part == "test"], "tail": d[d.is_tail]}
rows, preds = [], {}
for name, m in models.items():
    t0 = time.time(); m.fit(X, y); tfit = time.time() - t0
    for sn, s in sets.items():
        Xs = s[A.FEATURES].values
        t0 = time.time(); p = m.predict(Xs); tp = time.time() - t0
        yt, vt = s.y_vol.values, s.flood_volume_m3.values
        vp = np.expm1(np.clip(p, 0, None))
        flood_t, flood_p = (vt > 1).astype(int), (vp > 1).astype(int)
        flooded = vt > 1
        rows.append(dict(model=name, set=sn, n=len(s), R2_log=r2_score(yt, p),
                         RMSE_log=mean_squared_error(yt, p) ** .5,
                         R2_vol=r2_score(vt, vp), RMSE_vol_m3=mean_squared_error(vt, vp) ** .5,
                         NRMSE_vol=mean_squared_error(vt, vp) ** .5 / (vt.max() - vt.min()),
                         F1_flood=f1_score(flood_t, flood_p, zero_division=0) if flood_t.any() else np.nan,
                         AUC_flood=roc_auc_score(flood_t, p) if 0 < flood_t.mean() < 1 else np.nan,
                         miss_rate=float(((flood_t == 1) & (flood_p == 0)).sum() / max(flood_t.sum(), 1)),
                         false_alarm_rate=float(((flood_t == 0) & (flood_p == 1)).sum() / max((flood_t == 0).sum(), 1)),
                         fit_s=tfit, predict_us_per_case=1e6 * tp / len(s)))
        if sn == "test":
            preds[name] = p
    if name == "ANN_ensemble":
        mu, sd = m.predict(sets["test"][A.FEATURES].values, return_std=True)
        yt = sets["test"].y_vol.values
        rows.append(dict(model="ANN_ensemble", set="test", n=len(yt), note="PICP_95_ensemble_gaussian",
                         PICP=float((np.abs(yt - mu) <= 1.96 * sd).mean()), mean_sd_log=float(sd.mean())))
    if name == "GPR_Matern52_ARD":
        mu, sd = m.predict(sets["test"][A.FEATURES].values, return_std=True)
        yt = sets["test"].y_vol.values
        rows.append(dict(model=name, set="test", n=len(yt), note="PICP_95_gp",
                         PICP=float((np.abs(yt - mu) <= 1.96 * sd).mean()), mean_sd_log=float(sd.mean())))
        print("GPR learned kernel:", m.m.kernel_)
M = pd.DataFrame(rows)
M.to_csv(ROOT / "outputs" / "surrogate_metrics.csv", index=False)
out = sets["test"][["sample", "idf", "Tr_yr", "node", "CF", "flood_volume_m3"]].copy()
for k, p in preds.items():
    out["pred_" + k] = np.expm1(np.clip(p, 0, None))
out.to_csv(ROOT / "outputs" / "surrogate_predictions_test.csv", index=False)
with open(ROOT / "outputs" / "surrogate_ann.pkl", "wb") as f:
    pickle.dump(models["ANN_ensemble"], f)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"tail threshold CF >= {thr:.2f}")
print(M.drop(columns=["note"], errors="ignore").round(4).to_string(index=False))
# speed-up: SWMM on the user's machine = 1011 s wall, 4 workers, 72 000 node-results
sw = 1011 * 4 / 72000
ann_us = M[(M.model == "ANN_ensemble") & (M.set == "test") & M.R2_log.notna()].predict_us_per_case.iloc[0]
print(f"\nSWMM ~{sw*1e3:.0f} ms per node-case per core (user's PC); ANN ensemble {ann_us:.1f} us -> speed-up ~{sw/(ann_us*1e-6):,.0f}x (different hardware; indicative)")
