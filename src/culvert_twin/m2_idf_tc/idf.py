# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m2_idf_tc/idf.py
# ======================================================================================
"""Module 2: IDF formula, registry-table comparison and diagnostics.

I = a * Tr**m / (td + b)**c   (td in minutes, I in mm/hr), constants from config/registry.yaml.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def intensity(Tr, td_min, a, b, c, m):
    """Rainfall intensity (mm/hr) from the empirical IDF formula."""
    return a * np.asarray(Tr, dtype=float) ** m / (np.asarray(td_min, dtype=float) + b) ** c


def implied_duration(Tr, I_target, a, b, c, m):
    """Duration (min) that would make the formula return I_target at return period Tr."""
    return (a * np.asarray(Tr, dtype=float) ** m / np.asarray(I_target, dtype=float)) ** (1.0 / c) - b


def compare_table(reg) -> pd.DataFrame:
    """Registry 30-min table versus the registry's own formula."""
    p = reg.idf
    td = reg.design["duration_min"]
    rows = []
    for Tr, I_reg in sorted(p["table_30min_mm_hr"].items()):
        I_f = float(intensity(Tr, td, p["a"], p["b"], p["c"], p["m"]))
        rows.append({
            "Tr_yr": Tr,
            "registry_mm_hr": I_reg,
            "formula_mm_hr": round(I_f, 2),
            "ratio_registry_over_formula": round(I_reg / I_f, 3),
            "implied_td_min": round(float(implied_duration(Tr, I_reg, p["a"], p["b"], p["c"], p["m"])), 1),
        })
    return pd.DataFrame(rows)


def power_law_fit(reg) -> dict:
    """Fit ln I = alpha + k ln Tr to the registry table. If the table followed the formula, k would equal m."""
    t = reg.idf["table_30min_mm_hr"]
    Tr = np.array(sorted(t), dtype=float)
    I = np.array([t[int(x)] for x in Tr])
    k, alpha = np.polyfit(np.log(Tr), np.log(I), 1)
    resid = np.log(I) - (alpha + k * np.log(Tr))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.log(I) - np.log(I).mean()) ** 2))
    return {"exponent_k": float(k), "formula_m": reg.idf["m"], "r2": 1 - ss_res / ss_tot,
            "max_abs_residual_pct": float(np.max(np.abs(np.expm1(resid))) * 100)}


def design_intensity(reg, source: str = "registry_table", Tr: int | None = None) -> float:
    """Design intensity (mm/hr) for the baseline return period. source: 'registry_table' or 'formula'."""
    Tr = Tr or reg.design["return_period_yr"]
    if source == "registry_table":
        return float(reg.idf["table_30min_mm_hr"][Tr])
    if source == "formula":
        p = reg.idf
        return float(intensity(Tr, reg.design["duration_min"], p["a"], p["b"], p["c"], p["m"]))
    raise ValueError(f"unknown IDF source: {source}")


def aacra_cross_check(reg) -> pd.DataFrame:
    """Compare the registry 30-min table and formula with AACRA Bole values at 5/10/15 min (Tedla 2016).

    Physical check: intensity must not increase with duration, so the registry 30-min value must be
    <= the published 15-min value at the same return period. Only the return periods quoted are checked.
    """
    cc = reg.idf["cross_check_aacra_bole"]["mm_hr"]
    p = reg.idf
    rows = []
    for Tr, d in sorted(cc.items()):
        i15 = d[15]
        f15 = float(intensity(Tr, 15, p["a"], p["b"], p["c"], p["m"]))
        f30 = float(intensity(Tr, 30, p["a"], p["b"], p["c"], p["m"]))
        reg30 = p["table_30min_mm_hr"].get(Tr)
        rows.append({"Tr_yr": Tr, "AACRA_Bole_15min": i15, "registry_30min": reg30,
                     "registry_30min_le_15min": (reg30 <= i15) if reg30 is not None else None,
                     "formula_15min": round(f15, 1), "formula_30min": round(f30, 1),
                     "formula_15min_over_AACRA_15min": round(f15 / i15, 2)})
    return pd.DataFrame(rows)


def candidate_era(reg, Tr, d_min):
    """Intensity (mm/hr) from the UNVERIFIED candidate ERA A2 constants: i = C*Tr^m/(d+b)^e."""
    q = reg.idf["candidate_era_a2"]
    return q["C"] * np.asarray(Tr, dtype=float) ** q["m"] / (np.asarray(d_min, dtype=float) + q["b"]) ** q["e"]


def candidate_checks(reg) -> pd.DataFrame:
    """Test the candidate constants against (i) the registry table at 15 and 30 min, (ii) AACRA Bole 5/10/15 min (Tedla 2016)."""
    rows = []
    aac = {2: {5: 86, 10: 73, 15: 63}, 100: {5: 138, 10: 119, 15: 105}}
    for Tr, I_reg in sorted(reg.idf["table_30min_mm_hr"].items()):
        rows.append({"Tr_yr": Tr, "registry_table": I_reg,
                     "candidate_15min": round(float(candidate_era(reg, Tr, 15)), 1),
                     "candidate_30min": round(float(candidate_era(reg, Tr, 30)), 1),
                     "err_vs_table_at_15min_pct": round((float(candidate_era(reg, Tr, 15)) / I_reg - 1) * 100, 1),
                     "err_vs_table_at_30min_pct": round((float(candidate_era(reg, Tr, 30)) / I_reg - 1) * 100, 1),
                     "AACRA_15min": aac.get(Tr, {}).get(15),
                     "candidate_vs_AACRA_15min_pct": (round((float(candidate_era(reg, Tr, 15)) / aac[Tr][15] - 1) * 100, 1) if Tr in aac else None)})
    return pd.DataFrame(rows)


# ---------- user-supplied ERA table (UNVERIFIED source) ----------
from scipy.interpolate import RegularGridInterpolator


def _user_interp(reg):
    t = reg.idf["user_era_table"]
    d = np.log(np.array(t["durations_min"], dtype=float)); T = np.log(np.array(t["return_periods_yr"], dtype=float))
    I = np.log(np.array(t["mm_hr"], dtype=float))
    return RegularGridInterpolator((d, T), I, method="linear", bounds_error=True)


def user_table_intensity(reg, Tr, d_min):
    """Intensity (mm/hr) by bilinear interpolation of ln(i) over ln(duration) and ln(Tr) in the user-supplied table.
    Valid only inside 5-120 min and 2-100 yr (raises outside)."""
    f = _user_interp(reg)
    d, T = np.broadcast_arrays(np.asarray(d_min, dtype=float), np.asarray(Tr, dtype=float))
    return np.exp(f(np.column_stack([np.log(d.ravel()), np.log(T.ravel())]))).reshape(d.shape)


def user_table_checks(reg) -> pd.DataFrame:
    """Internal consistency and cross-source comparison of the user-supplied table at 15 and 30 min."""
    t = reg.idf["user_era_table"]; D = t["durations_min"]; R = t["return_periods_yr"]; G = np.array(t["mm_hr"], float)
    mono_d = bool(np.all(np.diff(G, axis=0) < 0)); mono_T = bool(np.all(np.diff(G, axis=1) > 0))
    p = reg.idf; rows = []
    aac = {2: {5: 86, 10: 73, 15: 63}, 100: {5: 138, 10: 119, 15: 105}}
    for d in (15, 30):
        for k, Tr in enumerate(R):
            i_user = G[D.index(d), k]
            rows.append({"duration_min": d, "Tr_yr": Tr, "user_table": i_user,
                         "registry_table30": p["table_30min_mm_hr"].get(Tr) if d == 30 else None,
                         "registry_formula": round(float(intensity(Tr, d, p["a"], p["b"], p["c"], p["m"])), 1),
                         "candidate_era": round(float(candidate_era(reg, Tr, d)), 1),
                         "AACRA_Bole_Tedla": aac.get(Tr, {}).get(d)})
    df = pd.DataFrame(rows); df.attrs["monotone_in_duration"] = mono_d; df.attrs["monotone_in_Tr"] = mono_T
    return df


def sherman_fit(reg) -> dict:
    """Fit ln i = ln a + m ln Tr - c ln(d + b) to the user table (all 42 points)."""
    from scipy.optimize import curve_fit
    t = reg.idf["user_era_table"]
    d = np.repeat(np.array(t["durations_min"], float), len(t["return_periods_yr"]))
    T = np.tile(np.array(t["return_periods_yr"], float), len(t["durations_min"])); I = np.array(t["mm_hr"], float).ravel()
    f = lambda X, a, m, b, c: np.log(a) + m * np.log(X[1]) - c * np.log(X[0] + b)
    p, _ = curve_fit(f, (d, T), np.log(I), p0=[500, .2, 10, .8], maxfev=20000)
    res = np.log(I) - f((d, T), *p)
    return {"a": p[0], "m": p[1], "b": p[2], "c": p[3], "rmse_log": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_err_pct": float(np.max(np.abs(np.expm1(res))) * 100)}
