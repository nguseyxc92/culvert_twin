"""Workflow diagram (Fig. 1) and corridor schematic (Fig. 2).
Author: Neguse Solomon Mekonnen, University of Padova. October 2026."""
import sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_figures")
out.mkdir(parents=True, exist_ok=True)
INK, FILL, ACC = "#1f2937", "#eef2f7", "#1d4e89"

def box(ax, x, y, w, h, text, fc=FILL, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=INK))

# ---- Figure 1: workflow
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
w, h = 1.75, 1.15
top = [("Inputs\nB, H, S, A, C\n3 IDF sources\nTc bracket", 0.1),
       ("Latin Hypercube\n2,000 draws per\nnode, IDF, Tr", 2.1),
       ("SWMM 5\ndynamic wave\n72,000 cases", 4.1),
       ("Surrogates\nANN, GPR,\npolynomial", 6.1),
       ("Reliability\nPf, Sobol,\nfragility", 8.1)]
for t, x in top: box(ax, x, 2.9, w, h, t, fs=8.5)
for i in range(4): arrow(ax, (top[i][1]+w, 3.47), (top[i+1][1], 3.47))
box(ax, 1.6, 0.9, 3.6, 1.0, "Checks\nSWMM vs Manning steady flow, continuity < 2 %\nheadroom test 0-2 m, 47 unit tests", fc="#fff7e6", fs=8)
box(ax, 6.1, 0.9, 3.75, 1.0, "Maintenance matrix\nstate x action: monitor, clean,\nclean now, upgrade", fc="#e8f3ec", fs=8.5)
arrow(ax, (4.6, 2.9), (4.2, 1.9)); arrow(ax, (8.97, 2.9), (8.2, 1.9))
ax.text(5, 4.35, "Workflow from corridor data to maintenance decision", ha="center", fontsize=11, weight="bold", color=INK)
fig.savefig(out/"fig_workflow.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: corridor schematic (upstream to downstream, not to scale)
nodes = [("N1", "Jemo 1", "1.5 x 1.2 m", "2,500 m, 1.20 %", "48 ha, C 0.80"),
         ("N2", "Jemo 2/3", "1.8 x 1.4 m", "2,250 m, 0.80 %", "65 ha, C 0.85"),
         ("N3", "German Sq.", "1.5 x 1.2 m", "2,000 m, 0.45 %", "42 ha, C 0.82"),
         ("N4", "Lancha outlet", "2.0 x 1.5 m", "2,000 m, 0.55 %", "58 ha, C 0.78")]
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
ax.plot([0.2, 9.8], [1.5, 1.5], color=ACC, lw=6, solid_capstyle="round", alpha=.55)
for k, (n, name, sec, run, cat) in enumerate(nodes):
    x = 0.4 + k * 2.4
    ax.add_patch(plt.Rectangle((x+0.55, 1.3), 0.5, 0.4, fc="white", ec=INK, lw=1.5))
    box(ax, x, 2.35, 1.65, 0.95, f"{n}  {name}\n{cat}", fs=8.5, bold=False)
    arrow(ax, (x+0.8, 2.35), (x+0.8, 1.72))
    ax.text(x+0.8, 0.95, f"box {sec}", ha="center", fontsize=8.5, color=INK)
    ax.text(x+0.8, 0.62, run, ha="center", fontsize=8.5, color=INK)
ax.text(0.2, 0.15, "Flow direction left to right. Schematic, not to scale; each culvert is modelled as one barrel and the nodes as independent.",
        fontsize=8, color="#555")
fig.savefig(out/"fig_corridor.png", dpi=200, bbox_inches="tight"); plt.close(fig)
