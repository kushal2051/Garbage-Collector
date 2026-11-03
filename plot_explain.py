"""Four panels explaining what the one-class detector is doing, from the arrays
detect.py saved. Reads out/viz_<slug>.npz, writes out/explain_<slug>.png.

PCA here is a viewing device only. The SVM is fit in the full 1,024-dim space;
these projections show where its flags landed, they do not produce them.

Usage:  python plot_explain.py [slug]      # slug defaults to "treats"
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

SLUG = sys.argv[1] if len(sys.argv) > 1 else "treats"
src = config.OUT / (f"viz_{SLUG}.npz")
config.require(src)
z = np.load(src, allow_pickle=True)

# validated 3-slot categorical palette (all-pairs, light surface) + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
BULK = "#b9b8b2"  # recessive gray for the unflagged mass

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
    }
)

pred = str(z["predicate"])
nu = float(z["nu"])
flag = z["flagged"]
score = z["score"]
n = len(flag)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
fig.suptitle(
    f"What the one-class detector sees   ·   {pred}, {n} edges, nu={nu:.2f}, "
    "RotatE 256 complex dims",
    x=0.012,
    ha="left",
    fontsize=12.5,
    fontweight="bold",
)

# ---- 1. the raw node space -------------------------------------------------
ax = axes[0, 0]
types = z["node_sample_types"]
pts = z["pca_nodes"]
uniq, counts = np.unique(types, return_counts=True)
top3 = uniq[np.argsort(-counts)][:3]
ax.scatter(*pts[~np.isin(types, top3)].T, s=2, c=BULK, alpha=0.35, linewidths=0, rasterized=True)
for col, tp in zip((BLUE, ORANGE, AQUA), top3, strict=False):
    m = types == tp
    ax.scatter(*pts[m].T, s=2, c=col, alpha=0.55, linewidths=0, rasterized=True)
    cx, cy = pts[m, 0].mean(), pts[m, 1].mean()
    ax.annotate(
        tp.replace("biolink:", ""),
        (cx, cy),
        color=INK,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        bbox=dict(fc=SURFACE, ec=col, lw=1.2, boxstyle="round,pad=0.25", alpha=0.95),
    )
ax.annotate(
    "other types", (0.97, 0.03), xycoords="axes fraction", ha="right", color=INK2, fontsize=8.5
)
ax.set_title("1 · The embedding space itself")
ax.set_xlabel("PC1 (%.0f%% var)" % (100 * z["evr_nodes"][0]))
ax.set_ylabel("PC2 (%.0f%% var)" % (100 * z["evr_nodes"][1]))

# ---- 2. where the flags land ----------------------------------------------
ax = axes[0, 1]
p = z["pca_concat"]
evr = z["evr_concat"]
ax.scatter(*p[~flag].T, s=5, c=BULK, alpha=0.5, linewidths=0, rasterized=True)
ax.scatter(*p[flag].T, s=9, c=ORANGE, alpha=0.9, linewidths=0, rasterized=True)
ax.annotate(
    f"flagged  n={int(flag.sum())}",
    (0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    color=ORANGE,
    fontweight="bold",
    fontsize=9,
)
ax.annotate(
    f"kept  n={int((~flag).sum())}",
    (0.97, 0.88),
    xycoords="axes fraction",
    ha="right",
    color=INK2,
    fontsize=9,
)
ax.set_title("2 · Flags in concat(h, t)")
ax.set_xlabel("PC1 (%.0f%% var)" % (100 * evr[0]))
ax.set_ylabel("PC2 (%.0f%% var)" % (100 * evr[1]))

# ---- 3. the diagnostic: is the score just node degree? --------------------
ax = axes[1, 0]
deg = z["min_degree"]
ax.scatter(deg[~flag], score[~flag], s=5, c=BULK, alpha=0.5, linewidths=0, rasterized=True)
ax.scatter(deg[flag], score[flag], s=9, c=ORANGE, alpha=0.9, linewidths=0, rasterized=True)
ax.axhline(0, color=INK2, lw=1, ls="--")
ax.annotate(
    "decision boundary",
    (0.02, 0.0),
    xycoords=("axes fraction", "data"),
    color=INK2,
    fontsize=8,
    va="bottom",
)
rho = np.corrcoef(np.argsort(np.argsort(deg)), np.argsort(np.argsort(score)))[0, 1]
ax.set_xscale("log")
ax.set_title(f"3 · Diagnostic: score vs node degree   (Spearman {rho:.2f})")
ax.set_xlabel("min(subject degree, object degree), log scale")
ax.set_ylabel("anomaly score")

# ---- 4. relation-specific distance, two predicates ------------------------
ax = axes[1, 1]
d1 = z["dist_l2"]
if "ref_dist_l2" in z.files:
    d2 = z["ref_dist_l2"]
    lo, hi = np.percentile(np.concatenate([d1, d2]), [0.2, 99.8])
    bins = np.linspace(lo, hi, 70)
    ax.hist(
        d1,
        bins=bins,
        color=BLUE,
        alpha=0.75,
        density=True,
        label=f"{pred.replace('biolink:', '')} (n={len(d1)})",
    )
    ax.hist(
        d2,
        bins=bins,
        color=ORANGE,
        alpha=0.65,
        density=True,
        label=f"{str(z['ref_label']).replace('biolink:', '')} (n={len(d2)})",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, labelcolor=INK)
    ax.set_title(
        "4 · RotatE distance, positives vs the evaluation negatives\n"
        "the overlap here is the detector's ceiling"
    )
else:
    ax.hist(d1, bins=70, color=BLUE, alpha=0.8, density=True)
    ax.set_title("4 · RotatE distance")
ax.set_xlabel(r"$\|h \circ r - t\|_2$")
ax.set_ylabel("density")

fig.tight_layout(rect=(0, 0.005, 1, 0.955))
dest = config.OUT / (f"explain_{SLUG}.png")
fig.savefig(dest, dpi=150, facecolor=SURFACE)
print(f"wrote {dest}")
