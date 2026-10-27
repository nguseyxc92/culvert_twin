"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/run_reliability.py

Usage: python scripts/run_reliability.py outputs/swmm_results.csv
Retrains the ANN ensemble on the training split, then writes to outputs/:
  reliability_pf.csv   surrogate Pf (100 000 LHS draws) vs SWMM Pf (2 000 draws, Wilson CI), per idf x Tr x node
  reliability_test.csv direct check on the held-out test samples (same inputs, SWMM vs surrogate)
  sobol.csv            S1/ST per scenario and node;  fragility.csv  Pf vs siltation beta
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m5_dataset import assemble as A
from culvert_twin.m6_surrogates.surrogates import ANNEnsemble
from culvert_twin.m7_reliability import reliability as R

csv = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "swmm_results.csv")
reg = load_registry(); d, _ = A.split(A.load(csv, reg))
tr_ = d[d.part == "train"]
model = ANNEnsemble().fit(tr_[A.FEATURES].values, tr_.y_vol.values)
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
scen = [(s, t) for s in ("registry_formula", "candidate", "user_table") for t in (10, 25, 50)]

pf, tst, sob = [], [], []
for src, tr in scen:
    for nd in reg.nodes:
        p, lo, hi, ev = R.failure_probability(model, reg, nd, tr, src)
        g = d[(d.idf == src) & (d.Tr_yr == tr) & (d.node == nd.id)]
        k, n = int(g.failed_swmm.sum()), len(g); slo, shi = sp.wilson(k, n)
        pf.append(dict(idf=src, Tr_yr=tr, node=nd.id, Pf_surrogate=p, Pf_sur_lo=lo, Pf_sur_hi=hi,
                       Pf_swmm=k / n, Pf_swmm_lo=slo, Pf_swmm_hi=shi, n_swmm=n,
                       sur_inside_swmm_CI=bool(slo <= p <= shi), E_flood_vol_m3=ev))
        gt = g[g.part == "test"]
        pr = np.expm1(np.clip(model.predict(gt[A.FEATURES].values), 0, None)) > 1
        tst.append(dict(idf=src, Tr_yr=tr, node=nd.id, n=len(gt), Pf_swmm=gt.failed_swmm.mean(), Pf_surrogate=pr.mean(),
                        abs_err=abs(gt.failed_swmm.mean() - pr.mean())))
        s = R.sobol(model, reg, nd, tr, src); s.insert(0, "node", nd.id); s.insert(0, "Tr_yr", tr); s.insert(0, "idf", src)
        sob.append(s)
pf, tst, sob = pd.DataFrame(pf), pd.DataFrame(tst), pd.concat(sob)
pf.to_csv(out / "reliability_pf.csv", index=False); tst.to_csv(out / "reliability_test.csv", index=False)
sob.to_csv(out / "sobol.csv", index=False)
betas = np.round(np.arange(0, 0.5001, 0.05), 2); fr = []
for src, tr in scen:
    for nd in reg.nodes:
        f = R.fragility(model, reg, nd, tr, src, betas); f.insert(0, "node", nd.id); f.insert(0, "Tr_yr", tr); f.insert(0, "idf", src); fr.append(f)
pd.concat(fr).to_csv(out / "fragility.csv", index=False)

pd.set_option("display.width", 220)
print("Pf: surrogate (100k LHS) vs SWMM (2000, 95% Wilson)  [registry_formula & candidate, Tr 10]")
print(pf[(pf.Tr_yr == 10)].round(3).drop(columns=["n_swmm", "E_flood_vol_m3"]).to_string(index=False))
print(f"\nsurrogate Pf inside SWMM 95% CI in {pf.sur_inside_swmm_CI.mean()*100:.0f}% of {len(pf)} cells; "
      f"max |Pf_sur - Pf_swmm| = {(pf.Pf_surrogate-pf.Pf_swmm).abs().max():.3f}")
print(f"held-out test samples: mean |dPf| = {tst.abs_err.mean():.4f}, max = {tst.abs_err.max():.4f}")
print("\nSobol (registry_formula, Tr 10), ST with 95% CI:")
print(sob[(sob.idf == "registry_formula") & (sob.Tr_yr == 10)].round(3).to_string(index=False))
