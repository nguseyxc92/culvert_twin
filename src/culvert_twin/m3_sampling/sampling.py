# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m3_sampling/sampling.py
# ======================================================================================
"""Module 3: uncertainty model, Latin Hypercube design, and algebraic evaluation (Rational + Manning).

Inputs (d = 1 + 3 * n_nodes = 13):
  u_tc            common fraction in [0,1]: Tc_k = Tc_velocity_k + u_tc * (Tc_kirpich_k - Tc_velocity_k)   (tc.py)
  n_k, beta_k, C_k  per node, independent, uniform on the ranges in registry.yaml (all ASSUMED, none verified)
Outputs per node: Q_peak, Q_cap (closed box, H_eff = H(1-beta)), CF = Q_peak/Q_cap, dQ = max(0, Q_peak-Q_cap), failed = CF > 1.
The Rational/Manning evaluation is the DETERMINISTIC BASELINE. The SWMM reference (Module 4) will replace `evaluate`
as the high-fidelity model; this file's design matrix is reused unchanged.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr, norm
from ..m1_baseline import baseline as bl
from ..m2_idf_tc import idf, tc as tcm


def input_names(reg):
    return ["u_tc"] + [f"{v}_{n.id}" for n in reg.nodes for v in ("n", "beta", "C")]


def build_design(reg, N=None, seed=None, optimization="random-cd") -> pd.DataFrame:
    """Physical-unit LHS design matrix. Each column has exactly one sample per stratum of width 1/N."""
    u = reg.uncertainty
    N = N or u["n_samples"]; seed = u["seed"] if seed is None else seed
    names = input_names(reg)
    unit = qmc.LatinHypercube(d=len(names), seed=seed, optimization=optimization).random(N)
    df = pd.DataFrame(unit, columns=names)
    out = pd.DataFrame({"u_tc": df["u_tc"]})
    for nd in reg.nodes:
        out[f"n_{nd.id}"] = u["manning_n"]["low"] + df[f"n_{nd.id}"] * (u["manning_n"]["high"] - u["manning_n"]["low"])
        out[f"beta_{nd.id}"] = u["beta"]["low"] + df[f"beta_{nd.id}"] * (u["beta"]["high"] - u["beta"]["low"])
        lo = nd.C - u["C"]["halfwidth"]; hi = min(nd.C + u["C"]["halfwidth"], u["C"]["cap"])
        out[f"C_{nd.id}"] = lo + df[f"C_{nd.id}"] * (hi - lo)
    return out


def _intensity(reg, source, tr, tc_min):
    if source == "candidate":
        return idf.candidate_era(reg, tr, tc_min)
    if source == "user_table":
        return idf.user_table_intensity(reg, tr, tc_min)
    if source == "registry_formula":
        p = reg.idf
        return idf.intensity(tr, tc_min, p["a"], p["b"], p["c"], p["m"])
    raise ValueError(source)


def node_inputs(reg, design: pd.DataFrame, nd, tr: int, idf_source: str):
    """Per-node storm inputs for every design row: (Tc [min], intensity [mm/hr]). Shared by the algebraic model and the SWMM runner."""
    tcd = tcm.tc_table(reg).set_index("node")
    lo, hi = tcd.loc[nd.id, "Tc_velocity_min"], tcd.loc[nd.id, "Tc_kirpich_min"]
    tc_min = lo + design["u_tc"].to_numpy() * (hi - lo)
    return tc_min, _intensity(reg, idf_source, tr, tc_min)


def evaluate(reg, design: pd.DataFrame, tr: int, idf_source: str) -> pd.DataFrame:
    """Deterministic algebraic model on every design row; returns a long table (sample x node)."""
    rows = []
    for nd in reg.nodes:
        tc_min, i = node_inputs(reg, design, nd, tr, idf_source)
        qpk = bl.rational_peak(design[f"C_{nd.id}"].to_numpy(), i, nd.area_ha)
        qcap = np.asarray(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, design[f"n_{nd.id}"].to_numpy(),
                                              design[f"beta_{nd.id}"].to_numpy(), "closed"), dtype=float)
        cf = np.where(qcap > 0, qpk / np.where(qcap > 0, qcap, 1), np.inf)
        rows.append(pd.DataFrame({"sample": np.arange(len(design)), "node": nd.id, "Tr_yr": tr, "idf": idf_source,
                                  "Tc_min": tc_min, "i_mm_hr": i, "n": design[f"n_{nd.id}"].to_numpy(),
                                  "beta": design[f"beta_{nd.id}"].to_numpy(), "C": design[f"C_{nd.id}"].to_numpy(),
                                  "Q_peak": qpk, "Q_cap": qcap, "CF": cf, "dQ": np.maximum(0, qpk - qcap),
                                  "failed": (qpk > qcap).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def wilson(k, n, conf=0.95):
    z = norm.ppf(0.5 + conf / 2); p = k / n
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (0.0 if lo < 1e-12 else lo), (1.0 if hi > 1 - 1e-12 else hi)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (idf_s, tr, node), g in res.groupby(["idf", "Tr_yr", "node"]):
        k, n = int(g["failed"].sum()), len(g); lo, hi = wilson(k, n); pf = k / n
        rows.append({"idf": idf_s, "Tr_yr": tr, "node": node, "Pf": round(pf, 4), "Pf_lo95": round(lo, 4), "Pf_hi95": round(hi, 4),
                     "beta_R": round(float(-norm.ppf(pf)), 2) if 0 < pf < 1 else (np.inf if pf == 0 else -np.inf),
                     "CF_median": round(float(g["CF"].median()), 2), "CF_p05": round(float(g["CF"].quantile(.05)), 2),
                     "CF_p95": round(float(g["CF"].quantile(.95)), 2), "E_dQ_m3s": round(float(g["dQ"].mean()), 2)})
    return pd.DataFrame(rows)


def convergence(reg, tr, idf_source, sizes=(250, 500, 1000, 2000, 4000)) -> pd.DataFrame:
    rows = []
    for N in sizes:
        r = evaluate(reg, build_design(reg, N), tr, idf_source)
        for node, g in r.groupby("node"):
            rows.append({"N": N, "node": node, "Pf": round(g["failed"].mean(), 4)})
    return pd.DataFrame(rows).pivot(index="N", columns="node", values="Pf")


def rank_sensitivity(reg, design, res_one: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of log(CF) with each input, per node (screening only; Sobol indices come with the surrogate)."""
    rows = []
    for nd in reg.nodes:
        g = res_one[res_one.node == nd.id].sort_values("sample")
        y = np.log(g["CF"].to_numpy())
        for v in ("u_tc", f"n_{nd.id}", f"beta_{nd.id}", f"C_{nd.id}"):
            rho = spearmanr(design[v].to_numpy(), y).statistic
            rows.append({"node": nd.id, "input": v.replace(f"_{nd.id}", ""), "spearman_rho": round(float(rho), 3)})
    return pd.DataFrame(rows).pivot(index="input", columns="node", values="spearman_rho")
