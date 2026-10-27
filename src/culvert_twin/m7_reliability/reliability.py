# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m7_reliability/reliability.py
# ======================================================================================
"""Module 7: reliability analysis with the trained surrogate.

  failure_probability : Pf = P[predicted flood volume > 1 m3] by Monte Carlo (LHS) over the uncertainty model
  sobol               : first-order (S1) and total (ST) Sobol indices, Saltelli/Jansen estimators, bootstrap CI,
                        on y = log1p(flood volume); inputs u_tc, n, beta, C of one node
  fragility           : Pf as a function of siltation beta (other inputs random)
No external Sobol package is used (SALib is not installable here); estimators are written out below.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from ..m3_sampling import sampling as sp

NAMES = ["u_tc", "n", "beta", "C"]


def physical(reg, nd, U):
    """Map unit-cube columns [u_tc, n, beta, C] to physical values (same ranges as sampling.build_design)."""
    u = reg.uncertainty
    lo_c = nd.C - u["C"]["halfwidth"]; hi_c = min(nd.C + u["C"]["halfwidth"], u["C"]["cap"])
    return dict(u_tc=U[:, 0],
                n=u["manning_n"]["low"] + U[:, 1] * (u["manning_n"]["high"] - u["manning_n"]["low"]),
                beta=u["beta"]["low"] + U[:, 2] * (u["beta"]["high"] - u["beta"]["low"]),
                C=lo_c + U[:, 3] * (hi_c - lo_c))


def features(reg, nd, ph, tr, src):
    tc_min, i = sp.node_inputs(reg, pd.DataFrame({"u_tc": ph["u_tc"]}), nd, tr, src)
    N = len(tc_min)
    cols = [np.full(N, nd.B_m), np.full(N, nd.H_m), np.full(N, nd.slope_registry), np.full(N, nd.area_ha),
            ph["n"], ph["beta"], ph["C"], tc_min, i]          # order = assemble.FEATURES
    return np.column_stack(cols)


def predict_logvol(model, reg, nd, U, tr, src):
    return np.clip(model.predict(features(reg, nd, physical(reg, nd, U), tr, src)), 0, None)


def failure_probability(model, reg, nd, tr, src, N=100_000, seed=1):
    U = qmc.LatinHypercube(d=4, seed=seed).random(N)
    y = predict_logvol(model, reg, nd, U, tr, src)
    k = int((np.expm1(y) > 1).sum()); lo, hi = sp.wilson(k, N)
    return k / N, lo, hi, float(np.expm1(y).mean())


def sobol(model, reg, nd, tr, src, N=16384, boot=200, seed=2):
    ab = qmc.Sobol(d=8, scramble=True, seed=seed).random(N)
    A, B = ab[:, :4], ab[:, 4:]
    fA, fB = (predict_logvol(model, reg, nd, M, tr, src) for M in (A, B))
    fAB = []
    for i in range(4):
        M = A.copy(); M[:, i] = B[:, i]; fAB.append(predict_logvol(model, reg, nd, M, tr, src))
    rng = np.random.default_rng(seed)

    def est(ix):
        a, b = fA[ix], fB[ix]; v = np.var(np.concatenate([a, b]))
        s1 = [np.mean(b * (f[ix] - a)) / v for f in fAB]
        st = [0.5 * np.mean((a - f[ix]) ** 2) / v for f in fAB]
        return np.array(s1), np.array(st)
    s1, st = est(np.arange(N))
    bs = [est(rng.integers(0, N, N)) for _ in range(boot)]
    s1b = np.array([b[0] for b in bs]); stb = np.array([b[1] for b in bs])
    return pd.DataFrame({"input": NAMES, "S1": s1, "S1_lo": np.percentile(s1b, 2.5, 0), "S1_hi": np.percentile(s1b, 97.5, 0),
                         "ST": st, "ST_lo": np.percentile(stb, 2.5, 0), "ST_hi": np.percentile(stb, 97.5, 0)})


def fragility(model, reg, nd, tr, src, betas, N=20_000, seed=3):
    U = qmc.LatinHypercube(d=4, seed=seed).random(N); rows = []
    u = reg.uncertainty
    for b in betas:
        Ub = U.copy(); Ub[:, 2] = (b - u["beta"]["low"]) / (u["beta"]["high"] - u["beta"]["low"])
        y = predict_logvol(model, reg, nd, Ub, tr, src)
        rows.append(dict(beta=b, Pf=float((np.expm1(y) > 1).mean()), E_vol_m3=float(np.expm1(y).mean())))
    return pd.DataFrame(rows)
