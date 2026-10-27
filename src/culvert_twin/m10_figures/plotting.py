# ======================================================================================
# Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
# Author : Neguse Solomon Mekonnen
# Date   : October 2026
# File   : src/culvert_twin/m10_figures/plotting.py
# ======================================================================================
"""Module 10 (partial): provisional figures for the baseline audit.

Palette: first categorical slots of the reference palette (blue, orange, aqua), validated for adjacent CVD separation.
Slot 3 (aqua) is below 3:1 contrast on white, so series are also distinguished by line style and every
series is in the legend. Text uses ink tokens, never series colours. One y-axis per panel.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..m2_idf_tc import idf
from ..m1_baseline import baseline as bl

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STAMP = "PROVISIONAL: registry inputs not yet verified (see Document 02, Part A)"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def fig_idf(reg, out_stem):
    p = reg.idf; td = reg.design["duration_min"]
    Tr_line = np.logspace(np.log10(2), np.log10(100), 100)
    f_line = idf.intensity(Tr_line, td, p["a"], p["b"], p["c"], p["m"])
    Trs = sorted(p["table_30min_mm_hr"]); I_reg = [p["table_30min_mm_hr"][t] for t in Trs]
    ratios = [r / float(idf.intensity(t, td, p["a"], p["b"], p["c"], p["m"])) for t, r in zip(Trs, I_reg)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200, facecolor=SURF); _style(ax)
    ax.plot(Tr_line, f_line, color=BLUE, lw=1.8, label="Registry IDF formula (a=520.4, b=14.5, c=0.78, m=0.19)")
    ax.plot(Trs, I_reg, color=ORANGE, lw=1.8, ls="--", marker="o", ms=6, mec=SURF, mew=1.5, label="Registry table, 30 min")
    ax.text(100, f_line[-1] - 6, "formula", color=INK2, fontsize=8.5, ha="right", va="top")
    ax.text(100, I_reg[-1] + 3, "registry table", color=INK2, fontsize=8.5, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_xticks(Trs); ax.set_xticklabels([str(t) for t in Trs])
    ax.minorticks_off()
    ax.set_xlabel("Return period (years)", color=INK2, fontsize=9); ax.set_ylabel("Rainfall intensity (mm/hr)", color=INK2, fontsize=9)
    ax.set_ylim(0, 125)
    ax.set_title(f"The registry IDF table sits {min(ratios)*100-100:.0f}–{max(ratios)*100-100:.0f}% above its own formula at 30 min",
                 loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.text(0.99, 0.01, STAMP, ha="right", va="bottom", fontsize=7, color=INK2, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor=SURF)
    plt.close(fig)


def fig_capacity(reg, out_stem):
    b = np.linspace(0, 0.8, 161); grid = np.array(reg.beta_grid)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), dpi=200, facecolor=SURF, sharex=True)
    i_tab = idf.design_intensity(reg, "registry_table"); i_frm = idf.design_intensity(reg, "formula")
    for ax, nd in zip(axes.ravel(), reg.nodes):
        _style(ax)
        qc = [np.asarray(bl.manning_capacity(nd.B_m, nd.H_m, nd.slope_registry, nd.n, b, per)) for per in ("closed", "open")]
        ax.plot(b, qc[0], color=ORANGE, lw=1.8, ls="--")
        ax.plot(b, qc[1], color=AQUA, lw=1.8, ls=":")
        ax.plot(grid, reg.reported["qcap_m3s"][nd.id], color=BLUE, lw=1.8, marker="o", ms=6, mec=SURF, mew=1.5)
        for i, ls, lab in ((i_tab, "-", "Q_peak, registry IDF"), (i_frm, "-.", "Q_peak, formula IDF")):
            q = bl.rational_peak(nd.C, i, nd.area_ha)
            ax.axhline(q, color=INK2, lw=0.9, ls=ls)
            ax.text(0.8, q, f"{lab} {q:.1f}", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_title(f"{nd.id} {nd.name}", loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Siltation ratio β", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Capacity or peak flow (m³/s)", color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=6, mec=SURF, label="Registry Q_cap"),
               plt.Line2D([], [], color=ORANGE, lw=1.8, ls="--", label="Recomputed Manning, closed box"),
               plt.Line2D([], [], color=AQUA, lw=1.8, ls=":", label="Recomputed Manning, open top")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Registry capacity starts lower than recomputed Manning values and falls more slowly with siltation",
                 x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.99, 0.005, STAMP, ha="right", va="bottom", fontsize=7, color=INK2, style="italic")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", facecolor=SURF)
    plt.close(fig)
