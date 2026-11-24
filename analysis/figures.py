#!/usr/bin/env python3
"""Fig. 4 and Fig. 5 for the MEDIC rerun, drawn with the first submission's figure code on the numbers computed by paper_numbers.py. Only the axis ranges are changed, to fit the new values.

    python3 analysis/figures.py   -> analysis/results/figures/
"""
import contextlib
import io
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(str(HERE / "paper_numbers.py"))
OUT = HERE / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S, summ, assign = P["S"], P["summ"], P["assign"]

rows = []
for t2 in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
    s = summ(assign(0.50, t2), True)
    rows.append(dict(arm="Stage-2", t1=0.50, t2=t2, qsat=s["q"], comp=s["comp"], lat=s["lat"], eng=s["j"]))
for t1 in (0.20, 0.35, 0.50, 0.65, 0.70, 0.80, 0.90):
    s = summ(assign(t1, 0.50), True)
    rows.append(dict(arm="Stage-1", t1=t1, t2=0.50, qsat=s["q"], comp=s["comp"], lat=s["lat"], eng=s["j"]))
sweep_df = pd.DataFrame(rows).drop_duplicates(subset=["arm", "t1", "t2"]).reset_index(drop=True)
sweep_df.to_csv(OUT / "table_fig4_threshold_operating_curve.csv", index=False)
pick = lambda k: {"qsat": S[k]["q"], "comp": S[k]["comp"]}  # noqa: E731
ref4 = {"Terminate@M5": pick("Fixed-Termination@M5"), "Terminate@Pi": pick("Fixed-Termination@Pi"),
        "Terminate@Jetson": pick("Fixed-Termination@Jetson"), "Oracle": pick("Quality-Cost Reference")}
deployed4 = pick("Consistency-Aware (Ours)")

# ---------------- Fig. 4 (cell 15)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0, "axes.labelsize": 9.0, "axes.linewidth": 0.8,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "figure.dpi": 200, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
OURS_FILL, OURS_EDGE, OURS_EDGE_H = "#BFE0CC", "#4FA176", "#2E7A50"
M5_FILL, M5_EDGE = "#BFD4EA", "#5E89B5"
PI_FILL, PI_EDGE = "#F6CFA8", "#C68A4B"
JET_FILL, JET_EDGE = "#F2B5BD", "#B86D78"
C_REF, C_ORC = "#F0F0F0", "#3B3B3B"

fig, ax = plt.subplots(figsize=(3.1, 2.5))
s2 = sweep_df[sweep_df.arm == "Stage-2"].sort_values("comp")
s1 = sweep_df[sweep_df.arm == "Stage-1"].sort_values("comp")
s1v = s1[~((np.isclose(s1.comp, deployed4["comp"])) & (np.isclose(s1.qsat, deployed4["qsat"])))]
s2v = s2[~((np.isclose(s2.comp, deployed4["comp"])) & (np.isclose(s2.qsat, deployed4["qsat"])))]

ref_style = {"Terminate@M5": (M5_FILL, M5_EDGE), "Terminate@Pi": (PI_FILL, PI_EDGE), "Terminate@Jetson": (JET_FILL, JET_EDGE)}


def curves(a, lw, ms, ref_s, dep_s):
    a.plot(s1.comp, s1.qsat, color=M5_EDGE, lw=lw, ls=(0, (3.2, 1.8)), alpha=0.9, zorder=3)
    a.scatter(s1v.comp, s1v.qsat, s=ms, facecolor="white", edgecolor=M5_EDGE, linewidth=1.05, zorder=4)
    a.plot(s2.comp, s2.qsat, color=JET_EDGE, lw=lw, alpha=0.9, zorder=3)
    a.scatter(s2v.comp, s2v.qsat, s=ms + 6, marker="^", facecolor="white", edgecolor=JET_EDGE, linewidth=1.05, zorder=4)
    for name, (fill, edge) in ref_style.items():
        r = ref4[name]
        a.scatter([r["comp"]], [r["qsat"]], s=ref_s, marker="s", facecolor=fill, edgecolor=edge, linewidth=0.95, alpha=0.96, zorder=2)
    a.scatter([ref4["Oracle"]["comp"]], [ref4["Oracle"]["qsat"]], s=ref_s + 4, marker="o", facecolor=C_REF, edgecolor=C_ORC, linewidth=1.2, zorder=5)
    a.scatter([deployed4["comp"]], [deployed4["qsat"]], s=dep_s, marker="o", facecolor=OURS_EDGE_H, edgecolor="white", linewidth=1.1, zorder=8)


curves(ax, 1.4, 21, 40, 42)

legend_handles = [
    Line2D([0], [0], color=M5_EDGE, lw=1.5, ls=(0, (3.2, 1.8)), marker="o", markersize=5.0, markerfacecolor="white", markeredgecolor=M5_EDGE, markeredgewidth=1.15, label="Stage 1 cutoff sweep"),
    Line2D([0], [0], color=JET_EDGE, lw=1.5, marker="^", markersize=5.4, markerfacecolor="white", markeredgecolor=JET_EDGE, markeredgewidth=1.15, label="Stage 2 cutoff sweep"),
]
leg = ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.015, -0.015), frameon=True,
                framealpha=0.9, facecolor="white", edgecolor="#CFCFCF", fontsize=6.6, borderpad=0.34,
                labelspacing=0.2, handlelength=1.6, handletextpad=0.45)
leg.get_frame().set_linewidth(0.7)

ann = [
    ("Fixed-Termination\n@M5", ref4["Terminate@M5"]["comp"], ref4["Terminate@M5"]["qsat"], -4, -6, "left", "top", M5_EDGE, False, 7.3),
    ("Fixed-Termination\n@Pi", ref4["Terminate@Pi"]["comp"], ref4["Terminate@Pi"]["qsat"], 2, -9, "left", "top", PI_EDGE, False, 7.3),
    ("Fixed-Termination\n@Jetson", ref4["Terminate@Jetson"]["comp"], ref4["Terminate@Jetson"]["qsat"], 0, 8, "center", "bottom", JET_EDGE, False, 7.3),
    ("Quality–Cost\nReference", ref4["Oracle"]["comp"], ref4["Oracle"]["qsat"], 0, -6, "center", "top", C_ORC, False, 7.3),
]
for txt, x, y, dx, dy, ha, va, col, bold, fs in ann:
    ax.annotate(txt, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fs,
                color=col, weight="bold" if bold else "normal", zorder=9)
ax.annotate("Deployed", xy=(deployed4["comp"], deployed4["qsat"]), xytext=(-34, 20), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.9, weight="bold", color=OURS_EDGE_H,
            arrowprops=dict(arrowstyle="-", color=OURS_EDGE_H, lw=0.8, shrinkA=1.0, shrinkB=3.5), zorder=9)

ax.set_xlim(-0.09, 1.04); ax.set_ylim(43.0, 93.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([50, 60, 70, 80, 90])
ax.set_xlabel("Composite system cost"); ax.set_ylabel("Q-Sat (%)")
ax.grid(True, linestyle=":", linewidth=0.45, color="#E2E2E2", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333333"); ax.spines["bottom"].set_color("#333333")
fig.tight_layout(pad=0.25)
fig.savefig(OUT / "fig4_threshold_operating_curve.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig4_threshold_operating_curve.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Fig. 5 (cell 18)
q_null, q_ours5 = P["qn"], P["q0"]
q_null_mean = float(np.mean(q_null))
pd.DataFrame({"q_sat_pct": q_null, "energy_j": P["en"]}).to_csv(OUT / "fig5_input_aware_permutation_null_samples.csv", index=False)
C_CDF, C_MEAN, C_OURS, C_OURS_E = "#8FB0CE", "#A9A9A9", "#DB8078", "#B85F57"
fig, ax = plt.subplots(figsize=(3.1, 2.5))
xs = np.sort(q_null)
vals, cnts = np.unique(xs, return_counts=True)
cdf = 100.0 * np.cumsum(cnts) / xs.size
pct_ours = 100.0 * float(np.mean(q_null < q_ours5))
tail = 100.0 - pct_ours

# The empirical CDF is 0 below the lowest draw and 100% above the highest. Draw it to both axis edges: in the rerun
# no draw reaches the proposed Q-Sat, and a curve stopping at the highest draw reads as a cut line.
x_lo, x_hi = xs.min() - 0.4, q_ours5 + 3.6
ax.plot(np.r_[x_lo, vals, x_hi], np.r_[0.0, cdf, 100.0], color=C_CDF, lw=1.7, solid_joinstyle="round", zorder=3)
pct_mean = 100.0 * float(np.mean(q_null < q_null_mean))
ax.plot([q_null_mean] * 2, [0, pct_mean], color=C_MEAN, lw=1.1, ls=(0, (3.2, 1.9)), zorder=4)
ax.text(q_null_mean + 0.2, 26, f"Random mean\n{q_null_mean:.1f}%", ha="left", va="center",
        fontsize=7.4, color="#6A6A6A", linespacing=1.22, zorder=6)
ax.plot([q_ours5] * 2, [0, pct_ours], color=C_OURS, lw=1.5, zorder=5)
ax.plot([q_ours5], [pct_ours], marker="D", ms=5.4, color=C_OURS, markerfacecolor=C_OURS,
        markeredgecolor=C_OURS_E, markeredgewidth=0.9, zorder=6)
ax.text(q_ours5 + 0.18, 69, f"Proposed  {q_ours5:.1f}%", ha="left", va="center", fontsize=7.6, color=C_OURS_E, weight="bold", zorder=6)
# no draw reached the proposed Q-Sat in the rerun, where "top 0.0%" would read as a rounding artefact
ax.text(q_ours5 + 0.18, 60, f"top {tail:.1f}%" if tail > 0 else f"above all\n{len(q_null):,} draws", ha="left", va="center",
        fontsize=7.4, color="#6A6A6A", linespacing=1.22, zorder=6)

ax.set_xlabel("Q-Sat of randomized assignments (%)")
ax.set_ylabel("Cumulative share (%)")
ax.set_xlim(x_lo, x_hi); ax.set_ylim(0, 106)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, linestyle=":", linewidth=0.45, color="#E2E2E2", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333333"); ax.spines["bottom"].set_color("#333333")
fig.tight_layout(pad=0.25)
fig.savefig(OUT / "fig5_input_aware_dotplot.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig5_input_aware_dotplot.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Fig. 4 and Fig. 5 written to {OUT}; Fig. 5: proposed {q_ours5:.2f}% is above {pct_ours:.2f}% of draws (top {tail:.2f}%)")
