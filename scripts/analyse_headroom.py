"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/analyse_headroom.py

Usage: python scripts/analyse_headroom.py outputs/swmm_headroom.csv
Writes outputs/headroom_pf.csv (Pf per headroom, IDF, Tr, node) and outputs/headroom_threshold.csv (CF at which half the cases flood).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m5_dataset import assemble as A
from culvert_twin.m3_sampling import sampling as sp

reg = load_registry(); d = pd.read_csv(sys.argv[1]); assert (d.error.fillna("") == "").all(), "error rows present"
nd = {n.id: n for n in reg.nodes}
d["A_ha"] = d.node.map(lambda k: nd[k].area_ha)
from culvert_twin.m1_baseline import baseline as bl
d["CF"] = [0.00278 * c * i * nd[k].area_ha / float(bl.manning_capacity(nd[k].B_m, nd[k].H_m, nd[k].slope_registry, n, b, "closed"))
           for c, i, k, n, b in zip(d.C, d.i_mm_hr, d.node, d.n, d.beta)]
g = d.groupby(["headroom_m", "idf", "Tr_yr", "node"]).failed_swmm.agg(["mean", "size"]).reset_index()
g["lo"], g["hi"] = zip(*[sp.wilson(int(round(m * n)), int(n)) for m, n in zip(g["mean"], g["size"])])
g = g.rename(columns={"mean": "Pf", "size": "n"}); g.round(4).to_csv(ROOT / "outputs" / "headroom_pf.csv", index=False)
rows = []
for h, gg in d.groupby("headroom_m"):
    gg = gg.sort_values("CF"); roll = gg.failed_swmm.rolling(300, center=True, min_periods=150).mean()
    rows.append(dict(headroom_m=h, CF_first_flood=gg.CF[gg.failed_swmm == 1].min(), CF_half_flooded=gg.CF[roll >= 0.5].min(),
                     Pf_all=gg.failed_swmm.mean(), median_flood_volume_m3=gg.loc[gg.flood_volume_m3 > 1, "flood_volume_m3"].median()))
t = pd.DataFrame(rows); t.round(4).to_csv(ROOT / "outputs" / "headroom_threshold.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False)); print()
print(g.pivot_table(index=["idf", "Tr_yr", "node"], columns="headroom_m", values="Pf").round(3).to_string())
