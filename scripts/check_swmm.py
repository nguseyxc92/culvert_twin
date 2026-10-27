# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/check_swmm.py
# ======================================================================================
"""Run on a computer with SWMM available (pip install pyswmm). Prints a PASS/FAIL checklist and writes outputs/swmm_check.csv.
Revised 24 Oct 2026 after diagnosis: steady surcharged flow is compared with Manning (not the transient peak), and the
Tc-consistent catchment is checked against the Rational peak."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m1_baseline import baseline as bl
from culvert_twin.m4_swmm import swmm_model as sm

reg = load_registry(); OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
nd = reg.node("N1"); rows = []


def case(beta=0.0, ratio=1.0, C=0.80, tc=30.0):
    qcap = float(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, nd.n, beta, "closed"))
    i = (ratio * qcap) / (0.00278 * C * nd.area_ha)       # rainfall giving Rational Q = ratio * Qcap
    return sm.Case("N1", nd.B_m, nd.H_m, nd.slope_registry, nd.length_m, nd.area_ha, nd.n, beta, C, tc, i), qcap


def run(c):
    ser = {}; r = sm.run_cases([c], series=ser)[0]; a = np.array(ser["N1"]); return r, a


def rec(name, ok, detail, info=False):
    res = "INFO" if info else ("PASS" if ok else "FAIL")
    rows.append({"check": name, "result": res, "detail": detail}); print(f"[{res}] {name}: {detail}")


for ratio in (0.3, 0.8):
    c, qcap = case(ratio=ratio, tc=60.0); r, a = run(c); q_r = ratio * qcap
    rec(f"no flooding at Q_rational = {ratio} Q_cap", r["flood_volume_m3"] < 1.0, f"flood volume {r['flood_volume_m3']} m3")
    rec(f"continuity errors at {ratio} Q_cap", abs(r["runoff_cont_err_pct"]) < 2 and abs(r["routing_cont_err_pct"]) < 2,
        f"runoff {r['runoff_cont_err_pct']:.2f} %, routing {r['routing_cont_err_pct']:.2f} %")
    rec(f"Tc-consistent catchment reaches the Rational peak (ratio {ratio})", 0.85 * q_r <= r["peak_flow_cms"] <= 1.05 * q_r,
        f"SWMM peak {r['peak_flow_cms']:.2f} vs Rational {q_r:.2f} m3/s ({r['peak_flow_cms']/q_r:.2f})")

c, qcap = case(ratio=3.0, tc=90.0); r, a = run(c)
plateau = a[(a[:, 0] >= 70) & (a[:, 0] <= 90), 1].mean()
rec("steady surcharged flow ~ Manning full-flow capacity", 0.95 * qcap <= plateau <= 1.06 * qcap, f"plateau {plateau:.2f} vs Manning {qcap:.2f} m3/s ({plateau/qcap:.3f})")
rec("transient peak vs steady flow (diagnosed as pipe fill / depressurisation spikes)", True, f"peak {a[:,1].max():.2f}, plateau {plateau:.2f} m3/s", info=True)
rec("flooding occurs when overloaded", r["flood_volume_m3"] > 1.0, f"flood volume {r['flood_volume_m3']:.0f} m3, peak flood rate {r['peak_flood_cms']:.2f} m3/s")
rec("mass balance: peak flood + steady conduit flow ~ peak inflow", abs(r["peak_flood_cms"] + plateau - 3.0 * qcap) < 0.1 * 3.0 * qcap,
    f"{r['peak_flood_cms']:.2f} + {plateau:.2f} vs {3.0*qcap:.2f} m3/s")

vols = []
base, _ = case(beta=0.0, ratio=1.0, tc=30.0)
for b in (0.0, 0.25, 0.5):
    cc = sm.Case(**{**base.__dict__, "beta": b}); vols.append(run(cc)[0]["flood_volume_m3"])
rec("flood volume non-decreasing with siltation", vols[0] <= vols[1] <= vols[2] and vols[2] > vols[0], f"beta 0/0.25/0.5 -> {[round(v) for v in vols]} m3")

pd.DataFrame(rows).to_csv(OUT / "swmm_check.csv", index=False)
print("\nOverall:", "ALL PASS" if all(r["result"] in ("PASS", "INFO") for r in rows) else "SOME FAILED: send outputs/swmm_check.csv")
