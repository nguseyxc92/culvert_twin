"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/run_maintenance.py

Usage: python scripts/run_maintenance.py   (reads outputs/fragility.csv from run_reliability.py)
Writes outputs/maintenance_thresholds.csv, maintenance_matrix.csv, maintenance_priority.csv and figures
outputs/fig_fragility.png, outputs/fig_maintenance_matrix.png
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from culvert_twin.m8_maintenance import matrix as M

out = ROOT / "outputs"; frag = pd.read_csv(out / "fragility.csv")
thr, mat, pri = M.build(frag)
thr.round(4).to_csv(out / "maintenance_thresholds.csv", index=False)
mat.round(4).to_csv(out / "maintenance_matrix.csv", index=False)
pri.round(4).to_csv(out / "maintenance_priority.csv", index=False)
NAMES = {"N1": "N1 Jemo 1", "N2": "N2 Jemo 2/3", "N3": "N3 German Sq.", "N4": "N4 Lancha outlet"}
COL = {"N1": "#2a78d6", "N2": "#eb6834", "N3": "#1baf7a", "N4": "#7a5cc8"}
pd.set_option("display.width", 220)

# ---- Fig: fragility curves, registry_formula, three return periods
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for a, tr in zip(ax, (10, 25, 50)):
    for n, g in frag[(frag.idf == "registry_formula") & (frag.Tr_yr == tr)].groupby("node"):
        g = g.sort_values("beta"); a.plot(g.beta, g.Pf, color=COL[n], lw=2, label=NAMES[n])
    a.axhline(M.P_ACT, color="#555", ls="--", lw=1); a.set_title(f"{tr}-year storm"); a.set_xlabel("siltation fraction β"); a.set_xlim(0, 0.5)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, fontsize=8, loc="upper left"); ax[0].set_ylabel("failure probability Pf"); ax[0].text(0.01, M.P_ACT + 0.015, "5 % action limit", fontsize=8, color="#555")
fig.suptitle("Fragility curves (registry IDF formula; surrogate, 20 000 draws per point)", fontsize=10); fig.tight_layout()
fig.savefig(out / "fig_fragility.png", dpi=200); plt.close(fig)

# ---- Fig: maintenance matrix, registry_formula, three return periods
code = {"MONITOR": 0, "CLEAN": 1, "CLEAN_NOW": 2, "UPGRADE": 3}
cm = ListedColormap(["#cfe8d8", "#f4e2a1", "#f2b79b", "#d98a8a"])
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for a, tr in zip(ax, (10, 25, 50)):
    s = mat[(mat.idf == "registry_formula") & (mat.Tr_yr == tr)]
    nodes = list(NAMES); states = [x[0] for x in M.STATES]
    Z = np.array([[code[s[(s.node == n) & (s.state == st)].action.iloc[0]] for st in states] for n in nodes])
    a.imshow(Z, cmap=cm, vmin=0, vmax=3, aspect="auto")
    for i, n in enumerate(nodes):
        for j, st in enumerate(states):
            r = s[(s.node == n) & (s.state == st)].iloc[0]; a.text(j, i, f"{r.action.replace('_', ' ')}\nPf {r.Pf:.2f}", ha="center", va="center", fontsize=7)
    a.set_xticks(range(4)); a.set_xticklabels(states, fontsize=8); a.set_yticks(range(4)); a.set_yticklabels([NAMES[n] for n in nodes], fontsize=8); a.set_title(f"{tr}-year storm")
fig.suptitle("Condition-based maintenance matrix (registry IDF formula; limit Pf = 5 %)", fontsize=10); fig.tight_layout()
fig.savefig(out / "fig_maintenance_matrix.png", dpi=200); plt.close(fig)

print("Thresholds, registry_formula (beta at which Pf reaches 5 % / 50 %; 'upgrade' = clean culvert already above 5 %):")
print(thr[thr.idf == "registry_formula"].round(2).to_string(index=False))
print("\nActions by state, registry_formula, Tr 10 / 25 / 50:")
print(mat[mat.idf == "registry_formula"].pivot_table(index=["Tr_yr", "node"], columns="state", values="action", aggfunc="first")[[x[0] for x in M.STATES]].to_string())
p = pri[(pri.idf == "registry_formula") & (pri.state == "Poor")]
print("\nBenefit of cleaning a 'Poor' (beta 0.325) node down to beta 0.05 (registry_formula):")
print(p.round(3).to_string(index=False))
print("\nUpgrade-needed count by idf source x Tr (of 4 nodes):")
print(thr.pivot_table(index="idf", columns="Tr_yr", values="upgrade_needed", aggfunc="sum"))
