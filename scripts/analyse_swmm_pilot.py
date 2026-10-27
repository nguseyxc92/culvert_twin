"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/analyse_swmm_pilot.py

Usage: python scripts/analyse_swmm_pilot.py path/to/swmm_results.csv
Checks integrity of a SWMM dataset and compares SWMM failure (flood volume > 1 m3)
with the algebraic criterion CF = Q_peak / Q_cap > 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m1_baseline import baseline as bl

reg = load_registry(); nodes = {n.id: n for n in reg.nodes}
d = pd.read_csv(sys.argv[1])
print("rows", len(d), "| error rows", int((d.error.fillna("") != "").sum()),
      "| max |continuity|", d[["runoff_cont_err_pct", "routing_cont_err_pct"]].abs().max().max())
d["Qpk"] = [0.00278 * c * i * nodes[k].area_ha for c, i, k in zip(d.C, d.i_mm_hr, d.node)]
d["Qcap"] = [float(bl.manning_capacity(nodes[k].B_m, nodes[k].H_m, nodes[k].slope_registry, n, b, "closed"))
             for k, n, b in zip(d.node, d.n, d.beta)]
d["CF"] = d.Qpk / d.Qcap
d["alg_fail"] = (d.CF > 1).astype(int)
print(pd.crosstab(d.alg_fail, d.failed_swmm, rownames=["algebraic"], colnames=["swmm"]))
bins = [0, .8, .9, 1, 1.1, 1.25, 1.5, 2, 3, 10]
print(d.groupby(pd.cut(d.CF, bins), observed=True).failed_swmm.agg(["size", "mean"]).round(3))
print("\nEffective CF threshold per node (smallest CF with SWMM flooding / CF where P(flood)=0.5):")
for k, g in d.groupby("node"):
    g = g.sort_values("CF"); roll = g.failed_swmm.rolling(60, center=True, min_periods=30).mean()
    c50 = g.CF[roll >= 0.5].min()
    print(f"  {k}: first flood CF={g.CF[g.failed_swmm == 1].min():.2f}, CF50={c50:.2f}")
print("\nPf by IDF source and return period (SWMM, all nodes):")
print(d.pivot_table(index="idf", columns="Tr_yr", values="failed_swmm", aggfunc="mean").round(3))
