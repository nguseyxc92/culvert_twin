# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m1_baseline/baseline.py
# ======================================================================================
"""Module 1: deterministic baseline audit (Rational Method + Manning) and reproduction checks.

Siltation follows the registry: H_eff = H(1 - beta), A_eff = B * H_eff.
Wetted perimeter is a modelling choice the registry leaves ambiguous, so both are provided:
  'open'   : P = B + 2*H_eff   (the formula printed in the registry; free surface at the silt line)
  'closed' : P = 2*B + 2*H_eff (flowing full under a lid; the physical box when full)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from ..m2_idf_tc.idf import design_intensity


# ---------- core hydraulics ----------
def rational_peak(C, i_mm_hr, area_ha):
    """Q (m3/s) = 0.00278 * C * i * A, with i in mm/hr and A in hectares."""
    return 0.00278 * C * i_mm_hr * area_ha


def manning_capacity(B, H, S, n, beta, perimeter="closed", barrels=1):
    """Full-section Manning capacity (m3/s) with siltation ratio beta (0 = clean, 1 = blocked)."""
    He = H * (1.0 - np.asarray(beta, dtype=float))
    A = B * He
    P = (B + 2.0 * He) if perimeter == "open" else (2.0 * B + 2.0 * He)
    Q = barrels * (1.0 / n) * A * (A / P) ** (2.0 / 3.0) * np.sqrt(S) / 1.0
    return np.where(He > 0, Q, 0.0)


def critical_beta(qcap_fn, q_peak, hi=0.999):
    """Smallest siltation ratio at which capacity drops to q_peak. 0.0 if overloaded even when clean."""
    if qcap_fn(0.0) <= q_peak:
        return 0.0
    if qcap_fn(hi) >= q_peak:
        return float("nan")
    return float(brentq(lambda b: qcap_fn(b) - q_peak, 0.0, hi))


def travel_time_min(length_m, q_cap_clean, B, H):
    """Indicative channel travel time at clean full-flow velocity (lower bound on channel time)."""
    return length_m / (q_cap_clean / (B * H)) / 60.0


# ---------- capacity sources ----------
def capacity_source(reg, node, source: str):
    """Return a function beta -> Q_cap (m3/s). source: 'registry' | 'closed' | 'open'."""
    if source == "registry":
        grid = np.array(reg.beta_grid)
        vals = np.array(reg.reported["qcap_m3s"][node.id])
        return lambda b: float(np.interp(b, grid, vals))
    if source in ("closed", "open"):
        return lambda b: float(manning_capacity(node.B_m, node.H_m, node.slope_registry, node.n, b, source))
    raise ValueError(source)


# ---------- reproduction checks ----------
def reproduce_registry(reg) -> pd.DataFrame:
    """Recompute every registry table and report PASS/FAIL with tolerances."""
    rows = []
    i10 = design_intensity(reg, "registry_table")
    beta = reg.design["beta_baseline"]
    for n in reg.nodes:
        rep = reg.reported["audit_10yr_beta030"][n.id]
        q = rational_peak(n.C, i10, n.area_ha)
        rows.append(_chk(n.id, "Q_peak (registry i)", rep["qpeak"], q, 0.005))
        rows.append(_chk(n.id, "bed slope", n.slope_registry, n.slope_from_dem, 0.005))
        sv = (rep["qpeak"] - rep["qcap"]) * reg.design["surcharge_duration_s"]
        rows.append(_chk(n.id, "surcharge volume", rep["surcharge_m3"], sv, 0.005))
        rows.append(_chk(n.id, "capacity factor", rep["cf"], rep["qpeak"] / rep["qcap"], 0.005))
        for k, b in enumerate(reg.beta_grid):
            rep_q = reg.reported["qcap_m3s"][n.id][k]
            for per in ("closed", "open"):
                calc = float(manning_capacity(n.B_m, n.H_m, n.slope_registry, n.n, b, per))
                rows.append(_chk(n.id, f"Q_cap beta={b:.2f} ({per})", rep_q, calc, 0.02))
    df = pd.DataFrame(rows)
    return df


def _chk(node, item, reported, recomputed, tol):
    err = (recomputed - reported) / reported
    return {"node": node, "item": item, "registry": round(float(reported), 4),
            "recomputed": round(float(recomputed), 4), "error_pct": round(float(err) * 100, 2),
            "tolerance_pct": tol * 100, "result": "PASS" if abs(err) <= tol else "FAIL"}


# ---------- hypothesis scan for the Q_cap table ----------
def qcap_hypotheses(reg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test candidate explanations for the registry Q_cap table.

    Returns (summary, implied_n). A hypothesis 'explains' the table only if errors are small
    over the whole beta grid, not just at beta = 0.
    """
    grid = np.array(reg.beta_grid)
    variants = {
        "closed box, n = 0.015": ("closed", None, False),
        "open top, n = 0.015": ("open", None, False),
        "closed box, n fitted per node (beta=0)": ("closed", "fit", False),
        "open top, n fitted per node (beta=0)": ("open", "fit", False),
        "closed box, twin/double barrels x2 (nodes 2,4)": ("closed", None, True),
    }
    rows, nrows = [], []
    for name, (per, fit, twin) in variants.items():
        errs = []
        for k, nd in enumerate(reg.nodes):
            rep = np.array(reg.reported["qcap_m3s"][nd.id])
            barrels = 2 if (twin and nd.id in ("N2", "N4")) else 1
            n_use = nd.n
            if fit == "fit":
                q0 = float(manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, 1.0, 0.0, per))
                n_use = q0 / rep[0]          # Q scales as 1/n, so this matches beta = 0 exactly
            calc = manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, n_use, grid, per, barrels)
            errs.extend(((calc - rep) / rep * 100).tolist())
        errs = np.array(errs)
        rows.append({"hypothesis": name, "max_abs_error_pct": round(float(np.max(np.abs(errs))), 1),
                     "rms_error_pct": round(float(np.sqrt(np.mean(errs ** 2))), 1),
                     "explains_table": bool(np.max(np.abs(errs)) <= 2.0)})
    for per in ("closed", "open"):
        for nd in reg.nodes:
            rep = np.array(reg.reported["qcap_m3s"][nd.id])
            q1 = manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, 1.0, grid, per)  # n = 1
            for b, r, q in zip(grid, rep, q1):
                nrows.append({"perimeter": per, "node": nd.id, "beta": b, "implied_n": round(float(q / r), 4)})
    return pd.DataFrame(rows), pd.DataFrame(nrows)


# ---------- audit scenarios (decision matrix) ----------
def audit_scenarios(reg, tr: int | None = None, beta: float | None = None) -> pd.DataFrame:
    """Audit under every combination of IDF source x capacity source."""
    beta = reg.design["beta_baseline"] if beta is None else beta
    rows = []
    for idf_src in ("registry_table", "formula"):
        i = design_intensity(reg, idf_src, tr)
        for cap_src in ("registry", "closed", "open"):
            for n in reg.nodes:
                qpk = rational_peak(n.C, i, n.area_ha)
                fn = capacity_source(reg, n, cap_src)
                qc = fn(beta)
                rows.append({"idf_source": idf_src, "capacity_source": cap_src, "node": n.id,
                             "i_mm_hr": round(i, 2), "Q_peak": round(qpk, 2),
                             f"Q_cap(beta={beta:.2f})": round(qc, 2),
                             "CF": round(qpk / qc, 2),
                             "critical_beta": round(critical_beta(fn, qpk), 3),
                             "overloaded_clean": bool(fn(0.0) < qpk)})
    return pd.DataFrame(rows)


def tc_indicator(reg) -> pd.DataFrame:
    """Indicative channel travel time (clean, full-flow, registry capacity) against the 30 min Rational limit."""
    rows = []
    for n in reg.nodes:
        q0 = reg.reported["qcap_m3s"][n.id][0]
        t = travel_time_min(n.length_m, q0, n.B_m, n.H_m)
        rows.append({"node": n.id, "area_ha": n.area_ha, "channel_travel_min": round(t, 1),
                     "area_over_50ha": n.area_ha > 50.0,
                     "note": "overland/inlet time must be added; Rational Method not to be used if Tc > 30 min"})
    return pd.DataFrame(rows)
