# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m8_maintenance/matrix.py
# ======================================================================================
"""Module 8: condition-based maintenance matrix from the fragility curves (Module 7).

Condition states by siltation fraction beta (share of the clear height blocked):
  Good 0-0.10 | Fair 0.10-0.25 | Poor 0.25-0.40 | Critical 0.40-0.50   (assumed bins, not from ERA/AACRA)
Action rule, given Pf(beta) of the state's mid-point and Pf at beta = 0 (clean culvert):
  Pf(0) >= P_ACT            -> UPGRADE   (cleaning cannot bring the node under the risk limit)
  Pf(beta) <  P_ACT         -> MONITOR   (routine inspection)
  P_ACT <= Pf(beta) < P_URG -> CLEAN     (schedule desilting)
  Pf(beta) >= P_URG         -> CLEAN_NOW (urgent desilting; upgrade if it recurs)
P_ACT = 0.05 (chosen by the author, an assumption); P_URG = 0.50.
No cost data exist, so nodes are ranked by risk reduction from cleaning, not by cost.
"""
import numpy as np
import pandas as pd

P_ACT, P_URG = 0.05, 0.50
STATES = [("Good", 0.00, 0.10), ("Fair", 0.10, 0.25), ("Poor", 0.25, 0.40), ("Critical", 0.40, 0.50)]
POST_CLEAN_BETA = 0.05


def _curve(g):
    g = g.sort_values("beta")
    return g.beta.to_numpy(), np.maximum.accumulate(g.Pf.to_numpy()), g.E_vol_m3.to_numpy()


def threshold(beta, pf, p):
    """Smallest beta at which Pf reaches p (linear interpolation); 0 if already above, NaN if never reached."""
    if pf[0] >= p: return 0.0
    if pf[-1] < p: return np.nan
    return float(np.interp(p, pf, beta))


def build(frag: pd.DataFrame):
    thr, mat, pri = [], [], []
    for (src, tr, node), g in frag.groupby(["idf", "Tr_yr", "node"]):
        b, pf, ev = _curve(g)
        pf0 = float(np.interp(0, b, pf)); upgrade = pf0 >= P_ACT
        thr.append(dict(idf=src, Tr_yr=tr, node=node, Pf_clean=pf0, beta_act=threshold(b, pf, P_ACT),
                        beta_urgent=threshold(b, pf, P_URG), upgrade_needed=upgrade))
        for name, lo, hi in STATES:
            mid = (lo + hi) / 2; p = float(np.interp(mid, b, pf))
            act = "UPGRADE" if upgrade else ("MONITOR" if p < P_ACT else ("CLEAN" if p < P_URG else "CLEAN_NOW"))
            mat.append(dict(idf=src, Tr_yr=tr, node=node, state=name, beta_mid=mid, Pf=p,
                            E_vol_m3=float(np.interp(mid, b, ev)), action=act))
        pc = float(np.interp(POST_CLEAN_BETA, b, pf)); ec = float(np.interp(POST_CLEAN_BETA, b, ev))
        for name, lo, hi in STATES[1:]:
            mid = (lo + hi) / 2
            pri.append(dict(idf=src, Tr_yr=tr, node=node, state=name, Pf_before=float(np.interp(mid, b, pf)),
                            Pf_after_clean=pc, dPf=float(np.interp(mid, b, pf)) - pc,
                            avoided_vol_m3=float(np.interp(mid, b, ev)) - ec))
    return pd.DataFrame(thr), pd.DataFrame(mat), pd.DataFrame(pri)
