"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : scripts/make_paper_figures.py

Usage: python scripts/make_paper_figures.py
Reads outputs/*.csv and writes the result figures used in the paper to outputs/paper_figures/
  fig_swmm_vs_algebraic.png   SWMM flood response against the algebraic capacity factor
  fig_surrogate_parity.png    surrogate accuracy (parity plot, error by model and test set)
  fig_pf_agreement.png        surrogate versus SWMM failure probability, 36 scenario-node cells
  fig_sobol.png               Sobol first-order and total-effect indices
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from culvert_twin.m0_config.config import load_registry
from culvert_twin.m5_dataset import assemble as A
from culvert_twin.m6_surrogates.surrogates import ANNEnsemble

OUT = ROOT / "outputs"; FIG = OUT / "paper_figures"; FIG.mkdir(exist_ok=True)
COL = {"N1": "#2a78d6", "N2": "#eb6834", "N3": "#1baf7a", "N4": "#7a5cc8"}
NAME = {"N1": "N1 Jemo 1", "N2": "N2 Jemo 2/3", "N3": "N3 German Sq.", "N4": "N4 Lancha outlet"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
reg = load_registry()
d, thr = A.split(A.load(OUT / "swmm_results.csv", reg))

# ---- Fig A: SWMM response versus algebraic CF
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
bins = np.arange(0.5, 4.01, 0.1); ctr = (bins[:-1] + bins[1:]) / 2
for n, g in d.groupby("node"):
    p = g.groupby(pd.cut(g.CF, bins), observed=False).failed_swmm.mean().values
    ax[0].plot(ctr, p, color=COL[n], lw=1.8, label=NAME[n])
ax[0].axvline(1.0, color="#555", ls="--", lw=1); ax[0].text(0.98, 0.5, "CF = 1\n(algebraic limit)", fontsize=8, color="#555", ha="right")
ax[0].set_xlabel("capacity factor CF = Q_peak / Q_cap"); ax[0].set_ylabel("share of cases with SWMM flooding")
ax[0].set_xlim(0.5, 3); ax[0].legend(frameon=False, fontsize=8, loc="center right"); ax[0].set_title("(a) Flooding onset", loc="left", fontsize=10)
f = d[d.CF > 1].copy(); f["v_alg"] = (f.Qpk_alg - f.Qcap_alg) * f.Tc_min * 60
f = f[f.flood_volume_m3 > 1].sample(6000, random_state=1)
ax[1].scatter(f.v_alg, f.flood_volume_m3, s=3, alpha=0.25, color="#2a78d6", lw=0)
lim = [30, 3e4]; ax[1].plot(lim, lim, color="#555", lw=1, ls="--"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("algebraic excess volume (Q_peak − Q_cap) × Tc  [m³]"); ax[1].set_ylabel("SWMM flood volume  [m³]")
ax[1].text(40, 1.6e4, "1:1", color="#555", fontsize=8); ax[1].set_title("(b) Flood volume", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig_swmm_vs_algebraic.png", dpi=300); plt.close(fig)

# ---- Fig B: surrogate accuracy
tr = d[d.part == "train"]; ann = ANNEnsemble().fit(tr[A.FEATURES].values, tr.y_vol.values)
te = d[d.part == "test"]; p = ann.predict(te[A.FEATURES].values)
M = pd.read_csv(OUT / "surrogate_metrics.csv"); M = M[M.R2_log.notna()]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(te.y_vol[~te.is_tail], p[~te.is_tail.values], s=3, alpha=0.25, color="#2a78d6", lw=0, label="test")
ax[0].scatter(te.y_vol[te.is_tail], p[te.is_tail.values], s=6, alpha=0.7, color="#eb6834", lw=0, label="tail (top 5 % CF)")
ax[0].plot([0, 10], [0, 10], color="#555", lw=1, ls="--"); ax[0].set_xlabel("SWMM  log(1 + flood volume)"); ax[0].set_ylabel("ANN ensemble prediction")
ax[0].legend(frameon=False, fontsize=8, loc="upper left"); ax[0].set_title("(a) Held-out parity", loc="left", fontsize=10)
models = ["ANN_ensemble", "GPR_Matern52_ARD", "PCE_deg3"]; lab = ["ANN ensemble", "GPR", "Polynomial"]; w = 0.26
for k, (s, c) in enumerate(zip(["val", "test", "tail"], ["#9bbbe8", "#2a78d6", "#eb6834"])):
    v = [M[(M.model == m) & (M.set == s)].RMSE_log.iloc[0] for m in models]
    ax[1].bar(np.arange(3) + (k - 1) * w, v, w, color=c, label=s)
    for x, y in zip(np.arange(3) + (k - 1) * w, v): ax[1].text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(lab); ax[1].set_ylabel("RMSE of log(1 + volume)")
ax[1].legend(frameon=False, fontsize=8, title="data set", title_fontsize=8); ax[1].set_title("(b) Error by model", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig_surrogate_parity.png", dpi=300); plt.close(fig)

# ---- Fig C: Pf agreement
R = pd.read_csv(OUT / "reliability_pf.csv")
fig, ax = plt.subplots(figsize=(5, 4.6))
for n, g in R.groupby("node"):
    ax.errorbar(g.Pf_swmm, g.Pf_surrogate, xerr=[g.Pf_swmm - g.Pf_swmm_lo, g.Pf_swmm_hi - g.Pf_swmm], fmt="o", ms=4, lw=0.8,
                color=COL[n], label=NAME[n])
ax.plot([0, 1], [0, 1], color="#555", lw=1, ls="--")
ax.set_xlabel("Pf from SWMM (2 000 draws, 95 % Wilson interval)"); ax.set_ylabel("Pf from surrogate (100 000 draws)")
ax.legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "fig_pf_agreement.png", dpi=300); plt.close(fig)

# ---- Fig D: Sobol
S = pd.read_csv(OUT / "sobol.csv"); S = S[(S.idf == "registry_formula") & (S.Tr_yr == 10)]
lab = {"u_tc": "Tc fraction", "n": "Manning n", "beta": "siltation β", "C": "runoff C"}
fig, ax = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for a, n in zip(ax, ["N1", "N2", "N3", "N4"]):
    g = S[S.node == n].set_index("input").loc[["beta", "u_tc", "n", "C"]]; x = np.arange(4)
    a.bar(x, g.ST, color=COL[n], alpha=0.85, label="total effect")
    a.errorbar(x, g.ST, yerr=[g.ST - g.ST_lo, g.ST_hi - g.ST], fmt="none", ecolor="#333", lw=1)
    a.scatter(x, g.S1, color="white", edgecolor="#222", zorder=3, s=22, label="first order")
    a.set_xticks(x); a.set_xticklabels([lab[i] for i in g.index], rotation=30, ha="right", fontsize=8); a.set_title(NAME[n], fontsize=9)
ax[0].set_ylabel("Sobol index"); ax[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig_sobol.png", dpi=300); plt.close(fig)
print("figures written to", FIG)

# ---- Fig E: IDF sources (paper version of the IDF comparison)
from culvert_twin.m3_sampling import sampling as sp
from culvert_twin.m1_baseline import baseline as bl
SRC = {"registry_formula": ("Registry formula", "#2a78d6"), "candidate": ("Candidate regional formula", "#eb6834"),
       "user_table": ("ERA table (interpolated)", "#1baf7a")}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
dur = np.linspace(10, 60, 26)
for k, (lab, c) in SRC.items():
    ax[0].plot(dur, [float(np.ravel(sp._intensity(reg, k, 10, t))[0]) for t in dur], color=c, lw=2, label=lab)
    trs = np.array([2, 5, 10, 25, 50, 100]); ax[1].plot(trs, [float(np.ravel(sp._intensity(reg, k, int(t), 30.0))[0]) for t in trs], color=c, lw=2, marker="o", ms=3.5, label=lab)
ax[0].set_xlabel("storm duration  [min]"); ax[0].set_ylabel("rainfall intensity  [mm/h]"); ax[0].set_title("(a) 10-year return period", loc="left", fontsize=10)
ax[1].set_xscale("log"); ax[1].set_xticks([2, 5, 10, 25, 50, 100]); ax[1].set_xticklabels(["2", "5", "10", "25", "50", "100"])
ax[1].set_xlabel("return period  [years]"); ax[1].set_title("(b) 30-minute duration", loc="left", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
for a in ax: a.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(FIG / "fig_idf_check.png", dpi=300); plt.close(fig)

# ---- Fig F: capacity of the closed box versus siltation, and the 10-year peak flow under each IDF source
fig, ax = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
b = np.linspace(0, 0.8, 81)
for a, nd in zip(ax.ravel(), reg.nodes):
    q15 = np.array([float(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, 0.015, x, "closed")) for x in b])
    q12 = np.array([float(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, 0.012, x, "closed")) for x in b])
    a.fill_between(b, q15, q12, color="#999", alpha=0.25, lw=0); a.plot(b, q15, color="#333", lw=1.6)
    for k, (lab, c) in SRC.items():
        i = float(np.ravel(sp._intensity(reg, k, 10, 30.0))[0]); q = float(bl.rational_peak(nd.C, i, nd.area_ha))
        a.axhline(q, color=c, lw=1.4, ls="--"); a.text(0.79, q, f"{q:.1f}", color=c, fontsize=8, va="bottom", ha="right")
    a.set_title(f"{nd.id} {nd.name}", loc="left", fontsize=9, fontweight="bold"); a.set_ylim(bottom=0)
for a in ax[1]: a.set_xlabel("siltation fraction β")
for a in ax[:, 0]: a.set_ylabel("flow  [m³/s]")
h = [plt.Line2D([], [], color="#333", lw=1.6), plt.Rectangle((0, 0), 1, 1, color="#999", alpha=0.25)] + [plt.Line2D([], [], color=c, ls="--") for _, c in SRC.values()]
fig.legend(h, ["Q_cap, n = 0.015", "Q_cap, n 0.012–0.015"] + [f"Q_peak, {l}" for l, _ in SRC.values()], ncol=5, frameon=False, fontsize=8, loc="lower center")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "fig_capacity_siltation.png", dpi=300); plt.close(fig)
print("IDF and capacity figures written")
