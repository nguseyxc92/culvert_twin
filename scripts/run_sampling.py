# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : scripts/run_sampling.py
# ======================================================================================
"""Run Module 3: LHS design, algebraic evaluation for all scenarios, Pf with 95% CI, convergence, rank sensitivity."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m3_sampling import sampling as sp

OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
reg = load_registry(); u = reg.uncertainty
t0 = time.time(); design = sp.build_design(reg); design.to_csv(OUT / "lhs_design.csv", index=False)
print(f"LHS design: N = {len(design)}, d = {design.shape[1]}, seed = {u['seed']}, built in {time.time()-t0:.1f} s")
print("assumed ranges:", {k: u[k] for k in ("manning_n", "beta", "C")})

allres = []
for src in u["scenarios"]["idf_sources"]:
    for tr in u["scenarios"]["return_periods_yr"]:
        allres.append(sp.evaluate(reg, design, tr, src))
res = pd.concat(allres, ignore_index=True); res.to_csv(OUT / "lhs_results_all_scenarios.csv", index=False)
summ = sp.summarise(res); summ.to_csv(OUT / "pf_summary.csv", index=False)
print("\n== FAILURE PROBABILITY Pf = P[Q_peak > Q_cap] (algebraic Rational/Manning baseline; PROVISIONAL) ==")
print(summ.to_string(index=False))

pv = summ.pivot_table(index=["idf", "Tr_yr"], columns="node", values="Pf")
print("\n== Pf matrix ==\n", pv.to_string())
rank = summ.assign(rank=summ.groupby(["idf", "Tr_yr"])["Pf"].rank(ascending=False, method="min")).pivot_table(index=["idf", "Tr_yr"], columns="node", values="rank")
print("\n== Rank of node by Pf (1 = most likely to fail); ties share a rank ==\n", rank.to_string())

for src in u["scenarios"]["idf_sources"]:
    conv = sp.convergence(reg, 10, src); conv.to_csv(OUT / f"convergence_{src}_10yr.csv")
    print(f"\n== CONVERGENCE of Pf with N ({src}, 10-yr) ==\n", conv.to_string())

one = res[(res.idf == "registry_formula") & (res.Tr_yr == 10)]
rs = sp.rank_sensitivity(reg, design, one); rs.to_csv(OUT / "rank_sensitivity_formula_10yr.csv")
print("\n== Spearman rho of log(CF) with each input (registry_formula, 10-yr) ==\n", rs.to_string())
one = res[(res.idf == "candidate") & (res.Tr_yr == 10)]
rs2 = sp.rank_sensitivity(reg, design, one); rs2.to_csv(OUT / "rank_sensitivity_candidate_10yr.csv")
print("\n== Spearman rho of log(CF) with each input (candidate, 10-yr) ==\n", rs2.to_string())
