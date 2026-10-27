# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/run_baseline.py
# ======================================================================================
"""Run modules 0-2: reproduction checks, hypothesis scan, audit scenarios. Writes outputs/*.csv and report.md."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m2_idf_tc import idf
from culvert_twin.m1_baseline import baseline as bl

OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
reg = load_registry()

idf_cmp = idf.compare_table(reg); idf_cmp.to_csv(OUT / "idf_check.csv", index=False)
fit = idf.power_law_fit(reg)
repro = bl.reproduce_registry(reg); repro.to_csv(OUT / "reproduction_check.csv", index=False)
hyp, implied = bl.qcap_hypotheses(reg); hyp.to_csv(OUT / "qcap_hypotheses.csv", index=False); implied.to_csv(OUT / "implied_n.csv", index=False)
scen = bl.audit_scenarios(reg); scen.to_csv(OUT / "audit_scenarios.csv", index=False)
tc = bl.tc_indicator(reg); tc.to_csv(OUT / "tc_indicator.csv", index=False)
xc = idf.aacra_cross_check(reg); xc.to_csv(OUT / "aacra_cross_check.csv", index=False)
adopted = scen[scen.capacity_source == reg.design["capacity_method"]]; adopted.to_csv(OUT / "adopted_capacity_audit.csv", index=False)

print("== STATUS (from registry) =="); [print(f"  {k:24s} {v}") for k, v in reg.status.items()]
print("\n== IDF: registry table vs registry formula (td = 30 min) ==\n", idf_cmp.to_string(index=False))
print(f"\n  table power-law exponent k = {fit['exponent_k']:.3f} (formula m = {fit['formula_m']}), R2 = {fit['r2']:.4f}, max residual {fit['max_abs_residual_pct']:.1f}%")
print("\n== REPRODUCTION SUMMARY ==")
print(repro.groupby(repro["item"].str.replace(r" beta=.*", "", regex=True))["result"].value_counts().unstack(fill_value=0).to_string())
print("\n== Q_cap HYPOTHESES (does any single explanation reproduce the table?) ==\n", hyp.to_string(index=False))
print("\n== implied n (closed box) by node and beta; constant => a pure roughness difference ==")
print(implied[implied.perimeter == "closed"].pivot(index="node", columns="beta", values="implied_n").to_string())
print("\n== AUDIT SCENARIOS, 10-yr, beta=0.30 (CF = Q_peak/Q_cap) ==")
piv = scen.pivot_table(index=["idf_source", "capacity_source"], columns="node", values="CF")
print(piv.to_string())
print("\n  critical beta (siltation at which capacity = Q_peak; 0 = overloaded even when clean):")
print(scen.pivot_table(index=["idf_source", "capacity_source"], columns="node", values="critical_beta").to_string())
print(f"\n== ADOPTED CAPACITY METHOD: {reg.design['capacity_method']} (full-flow box, soffit wetted) ==")
print(adopted.to_string(index=False))
print("\n== AACRA (Bole, Tedla 2016) cross-check: 30-min intensity must be <= 15-min intensity ==\n", xc.to_string(index=False))
cc = idf.candidate_checks(reg); cc.to_csv(OUT / "candidate_era_check.csv", index=False)
print("\n== CANDIDATE ERA A2 constants (UNVERIFIED, pasted 24 Oct 2026) vs registry table and AACRA ==\n", cc.to_string(index=False))
import numpy as np
rows = []
for d in (15, 30):
    i = float(idf.candidate_era(reg, 10, d))
    for n in reg.nodes:
        qpk = bl.rational_peak(n.C, i, n.area_ha); fn = bl.capacity_source(reg, n, "closed")
        rows.append({"duration_min": d, "i_10yr": round(i, 1), "node": n.id, "Q_peak": round(qpk, 2), "Q_cap_closed(b=0.30)": round(fn(0.30), 2),
                     "CF": round(qpk / fn(0.30), 2), "critical_beta": round(bl.critical_beta(fn, qpk), 3)})
cand_audit = pd.DataFrame(rows); cand_audit.to_csv(OUT / "candidate_era_audit.csv", index=False)
print("\n== AUDIT with candidate ERA constants, 10-yr, closed-box capacity ==\n", cand_audit.to_string(index=False))
from culvert_twin.m2_idf_tc import tc as tcm
tct = tcm.tc_table(reg); tct.to_csv(OUT / "tc_table.csv", index=False)
print("\n== TIME OF CONCENTRATION: two brackets (reach length only; catchment flow path not in registry) ==\n", tct.to_string(index=False))
at = pd.concat([tcm.audit_at_tc(reg, 10, source="candidate"), tcm.audit_at_tc(reg, 10, source="registry_formula")])
at.to_csv(OUT / "audit_at_tc.csv", index=False)
print("\n== AUDIT with duration = Tc (10-yr, closed box, beta=0.30) ==\n", at.to_string(index=False))
ut = idf.user_table_checks(reg); ut.to_csv(OUT / "user_idf_table_check.csv", index=False)
sf = idf.sherman_fit(reg)
print("\n== USER-SUPPLIED ERA IDF TABLE (xlsx; source page unconfirmed) vs other sources ==\n", ut.to_string(index=False))
print(f"  monotone in duration: {ut.attrs['monotone_in_duration']}, monotone in Tr: {ut.attrs['monotone_in_Tr']}")
print(f"  Sherman fit i = a Tr^m/(d+b)^c: a={sf['a']:.1f} m={sf['m']:.3f} b={sf['b']:.2f} c={sf['c']:.3f}  (RMSE ln i = {sf['rmse_log']:.3f}, max error {sf['max_abs_err_pct']:.1f} %)")
fp = tcm.flow_path_table(reg); fp.to_csv(OUT / "flow_path_check.csv", index=False)
print("\n== USER-SUPPLIED GENERIC FLOW PATH (xlsx) ==\n", fp.to_string(index=False))
print(f"  total stated {fp.attrs['total_stated']:.1f} min; with kinematic-wave sheet flow {fp.attrs['total_with_kinematic_sheet']:.1f} min")
print("\n== Rational-Method applicability (indicative) ==\n", tc.to_string(index=False))

# ---- provisional figures ----
from culvert_twin.m10_figures import plotting
plotting.fig_idf(reg, str(OUT / "fig04_idf_check"))
plotting.fig_capacity(reg, str(OUT / "fig05_capacity_vs_siltation"))
print("\nfigures written to", OUT)
