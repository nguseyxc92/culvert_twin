# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m2_idf_tc/tc.py
# ======================================================================================
"""Module 2b: time of concentration (Tc) per node, and the Tc-consistent design intensity.

Two independent estimates bracket Tc because the registry gives only the culvert reach length, not the
catchment flow-path length:
  * Kirpich (1940):  Tc[min] = 0.0195 * L[m]**0.77 * S**-0.385   (calibrated on small rural catchments;
    a published standard form, applicability to a concrete urban corridor is NOT verified)
  * channel-velocity: Tc = L / V_full, V_full = Q_cap(clean, closed box) / (B*H)
    (lower bound: concrete conduit flowing full; ignores overland and inlet time)
Rational-Method rule quoted in the AfCAP/ERA workshop record: do not use if Tc > 30 min.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..m1_baseline import baseline as bl
from . import idf


def kirpich_min(L_m, S):
    return 0.0195 * np.asarray(L_m, dtype=float) ** 0.77 * np.asarray(S, dtype=float) ** -0.385


def velocity_min(node) -> float:
    q0 = float(bl.manning_capacity(node.B_m, node.H_m, node.slope_registry, node.n, 0.0, "closed"))
    return node.length_m / (q0 / (node.B_m * node.H_m)) / 60.0


def tc_table(reg) -> pd.DataFrame:
    rows = []
    for n in reg.nodes:
        tk, tv = float(kirpich_min(n.length_m, n.slope_registry)), velocity_min(n)
        rows.append({"node": n.id, "L_m": n.length_m, "S": n.slope_registry, "area_ha": n.area_ha,
                     "Tc_velocity_min": round(tv, 1), "Tc_kirpich_min": round(tk, 1),
                     "kirpich_over_30min": tk > 30, "area_over_50ha": n.area_ha > 50})
    return pd.DataFrame(rows)


def audit_at_tc(reg, tr=10, beta=None, source="candidate") -> pd.DataFrame:
    """CF and critical beta using duration = Tc (both bracket values). source: 'candidate' (UNVERIFIED ERA A2) or 'registry_formula'."""
    beta = reg.design["beta_baseline"] if beta is None else beta
    rows = []
    for n in reg.nodes:
        fn = bl.capacity_source(reg, n, "closed")
        for label, tc in (("velocity", velocity_min(n)), ("kirpich", float(kirpich_min(n.length_m, n.slope_registry)))):
            if source == "candidate":
                i = float(idf.candidate_era(reg, tr, tc))
            else:
                p = reg.idf; i = float(idf.intensity(tr, tc, p["a"], p["b"], p["c"], p["m"]))
            q = bl.rational_peak(n.C, i, n.area_ha)
            rows.append({"idf": source, "Tr_yr": tr, "node": n.id, "Tc_basis": label, "Tc_min": round(tc, 1), "i_mm_hr": round(i, 1),
                         "Q_peak": round(q, 2), f"Q_cap(b={beta:.2f})": round(fn(beta), 2), "CF": round(q / fn(beta), 2),
                         "critical_beta": round(bl.critical_beta(fn, q), 3)})
    return pd.DataFrame(rows)


def flow_path_table(reg, i_mm_hr: float = 83.0) -> pd.DataFrame:
    """Author-supplied generic flow path (xlsx): recompute travel times and cross-check the sheet-flow segment.
    Sheet flow cross-check: kinematic-wave (Morgali & Linsley) Tt[min] = 6.94 (L n)^0.6 / (i^0.4 S^0.3), L in m, i in mm/hr.
    The sheet states 11.1 min from an assumed 0.15 m/s, which is not derived from L, n, S or rainfall."""
    rows = []
    for sgm in reg.tc_flow_path["segments"]:
        t_calc = sgm["length_m"] / sgm["v_m_s"] / 60.0
        row = {"segment": sgm["name"], "L_m": sgm["length_m"], "S": sgm["slope"], "v_m_s": sgm["v_m_s"],
               "stated_min": sgm["stated_min"], "recomputed_min": round(t_calc, 1), "kinematic_min": None}
        if sgm["kind"] == "sheet":
            row["kinematic_min"] = round(6.94 * (sgm["length_m"] * sgm["n"]) ** 0.6 / (i_mm_hr ** 0.4 * sgm["slope"] ** 0.3), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["total_stated"] = float(df["stated_min"].sum())
    df.attrs["total_with_kinematic_sheet"] = float(df["recomputed_min"].sum() - df.loc[0, "recomputed_min"] + df.loc[0, "kinematic_min"])
    return df
