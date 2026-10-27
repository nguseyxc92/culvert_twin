# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/diagnose_swmm.py
# ======================================================================================
"""Diagnose why SWMM's peak conduit flow (7.35 m3/s) exceeded the Manning full-flow capacity (6.32 m3/s) for Node 1.
Run on the computer that has pyswmm:   py scripts/diagnose_swmm.py
Send back outputs/swmm_diagnose.csv and outputs/swmm_series_overloaded.csv.
Hypotheses tested: (A) transient surge versus quasi-steady plateau, (B) head from the inlet headroom, (C) routing step,
(D) sensitivity of the SWMM response to the assumed overland flow-path length (affects lag and peak attenuation)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m1_baseline import baseline as bl
from culvert_twin.m4_swmm import swmm_model as sm

reg = load_registry(); OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
nd = reg.node("N1"); QCAP = float(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, nd.n, 0.0, "closed"))
rows = []


def mk(ratio, tc, C=0.80, beta=0.0):
    i = ratio * QCAP / (0.00278 * C * nd.area_ha)
    return sm.Case("N1", nd.B_m, nd.H_m, nd.slope_registry, nd.length_m, nd.area_ha, nd.n, beta, C, tc, i)


def stats(ser, tc, qref):
    t = np.array([x[0] for x in ser]); q = np.array([x[1] for x in ser]); d = np.array([x[2] for x in ser])
    plateau = q[(t >= tc - 20) & (t <= tc)].mean()
    return {"peak_flow": q.max(), "t_peak_min": t[q.argmax()], "plateau_flow_last20min": plateau, "max_depth": d.max(),
            "peak_over_qref": q.max() / qref, "plateau_over_qref": plateau / qref}


def run(label, case, opt=None, step_s=10, qref=QCAP, save=None):
    ser = {}; r = sm.run_cases([case], opt, step_s=step_s, series=ser)[0]
    st = stats(ser["N1"], case.tc_min, qref); rows.append({"test": label, "Q_ref": round(qref, 3), **{k: round(float(v), 3) for k, v in st.items()},
                                                           "flood_volume_m3": round(r["flood_volume_m3"]), "peak_flood_cms": round(r["peak_flood_cms"], 3)})
    print(rows[-1])
    if save: pd.DataFrame(ser["N1"], columns=["t_min", "flow_cms", "depth_m", "flooding_cms"]).to_csv(OUT / save, index=False)


print(f"Manning closed-box full-flow capacity Q_cap = {QCAP:.3f} m3/s")
run("A. baseline overloaded 3x Qcap, Tc=90", mk(3.0, 90.0), save="swmm_series_overloaded.csv")
for h in (0.1, 1.0, 3.0):
    run(f"B. inlet headroom {h} m", mk(3.0, 90.0), opt=sm.Options(inlet_headroom_m=h))
for st in (2, 10, 30):
    run(f"C. routing step {st} s", mk(3.0, 90.0), opt=sm.Options(routing_step_s=st), step_s=st)
q03 = 0.3 * QCAP
for L in (100, 300, 1000):
    run(f"D. overland flow path {L} m, storm = 0.3 Qcap for 60 min", mk(0.3, 60.0), opt=sm.Options(flow_path_m=float(L)), qref=q03)
pd.DataFrame(rows).to_csv(OUT / "swmm_diagnose.csv", index=False)
print("\nwritten", OUT / "swmm_diagnose.csv")
