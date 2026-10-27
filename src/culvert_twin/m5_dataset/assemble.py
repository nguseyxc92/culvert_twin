"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : src/culvert_twin/m5_dataset/assemble.py

Module 5: turn swmm_results.csv into a surrogate-ready table and splits.

Surrogate inputs (all physical, IDF-source independent):
  node geometry (B, H, S, A_ha), Manning n, siltation beta, runoff C, Tc_min, i_mm_hr
Targets: flood_volume_m3 (modelled as log1p), peak_flood_cms.
Splits are made by design sample id (so the same LHS draw never sits in two splits):
  train 70 % / validation 15 % / test 15 %, plus a TAIL test set = test rows whose
  algebraic CF is in the top 5 % of the whole dataset (extrapolation to extremes).
"""
import numpy as np, pandas as pd

FEATURES = ["B_m", "H_m", "S", "A_ha", "n", "beta", "C", "Tc_min", "i_mm_hr"]


def load(csv, reg):
    d = pd.read_csv(csv)
    assert (d.error.fillna("") == "").all(), "dataset contains error rows"
    nd = {n.id: n for n in reg.nodes}
    d["B_m"] = d.node.map(lambda k: nd[k].B_m)
    d["H_m"] = d.node.map(lambda k: nd[k].H_m)
    d["S"] = d.node.map(lambda k: nd[k].slope_registry)
    d["A_ha"] = d.node.map(lambda k: nd[k].area_ha)
    d["Qpk_alg"] = 0.00278 * d.C * d.i_mm_hr * d.A_ha
    from culvert_twin.m1_baseline import baseline as bl
    d["Qcap_alg"] = [float(bl.manning_capacity(b, h, s, n, be, "closed"))
                     for b, h, s, n, be in zip(d.B_m, d.H_m, d.S, d.n, d.beta)]
    d["CF"] = d.Qpk_alg / d.Qcap_alg
    d["y_vol"] = np.log1p(d.flood_volume_m3)
    return d


def split(d, seed=20261024):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(d["sample"].unique())); rng.shuffle(ids)
    n = len(ids); tr, va = ids[: int(.7 * n)], ids[int(.7 * n): int(.85 * n)]
    part = np.where(d["sample"].isin(tr), "train", np.where(d["sample"].isin(va), "val", "test"))
    d = d.assign(part=part)
    thr = d.CF.quantile(0.95)
    d["is_tail"] = (d.part == "test") & (d.CF >= thr)
    return d, thr
