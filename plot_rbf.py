"""What the RBF one-class detector flags, for the rbf-only replication run.

Refits the selected model (rbf, nu=0.2, gamma='scale') on the same features the
run used -- L2-normalised concat(h, t), then standardised -- so we have the
decision function, not just the +/-1 labels the run wrote out. PCA is a viewing
device only; the SVM is fit in the full 1024-dim space.

Usage:  python plot_rbf.py [run_dir_name]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config

EMB = config.EMB
TYPES = config.NODE_TYPES
RUN = sys.argv[1] if len(sys.argv) > 1 else "original_treats_rbf"
NU, GAMMA, KERNEL = 0.2, "scale", "rbf"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
BULK = "#b9b8b2"
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

# ----------------------------------------------------------------- inputs
res = pd.read_csv(config.OUT / RUN / "OneClassSVM_Results_treats.csv")
nodes = np.load(EMB / "node_embeddings.npy", mmap_mode="r")
idx = pd.read_csv(EMB / "node_index.tsv", sep="\t")
row_of = dict(zip(idx.curie, idx.node_id, strict=False))
name_of = dict(zip(idx.curie, idx.name, strict=False))
deg = pd.read_csv(config.DEGREES_TSV, sep="\t")
degree = dict(zip(deg.curie, deg.in_degree + deg.out_degree, strict=False))
types = pd.read_csv(TYPES, sep="\t", usecols=["curie", "selected_type"])
type_of = dict(zip(types.curie, types.selected_type, strict=False))

res["s_row"] = res.subject.map(row_of)
res["o_row"] = res.object.map(row_of)
res = res.dropna(subset=["s_row", "o_row"]).reset_index(drop=True)

h = np.asarray(nodes[res.s_row.values.astype(int)], dtype=np.float64)
t = np.asarray(nodes[res.o_row.values.astype(int)], dtype=np.float64)
X = np.hstack([h, t])
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # L2 row-normalise
Xs = StandardScaler().fit_transform(X)

print(f"refitting {KERNEL} nu={NU} gamma={GAMMA} on {Xs.shape}", flush=True)
svm = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA).fit(Xs)
score = svm.decision_function(Xs)
flag = score < 0
print(
    f"flagged {flag.sum()} / {len(flag)} ({100 * flag.mean():.1f}%)  "
    f"[run wrote {(res.predictions == -1).sum()}]"
)

res["score"] = score
res["flag"] = flag
res["sdeg"] = res.subject.map(degree).fillna(0)
res["odeg"] = res.object.map(degree).fillna(0)
res["min_degree"] = res[["sdeg", "odeg"]].min(axis=1)
rho = spearmanr(res.score, res.min_degree).statistic

pcs = PCA(n_components=2).fit(Xs)
P = pcs.transform(Xs)
evr = pcs.explained_variance_ratio_

# ----------------------------------------------------------------- figure
fig = plt.figure(figsize=(19.5, 11.6))
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1, 1.08],
    hspace=0.30,
    wspace=0.22,
    left=0.045,
    right=0.985,
    top=0.885,
    bottom=0.05,
)
fig.suptitle(
    "What the RBF one-class detector flags   ·   biolink:treats, "
    f"{len(res)} edges, kernel=rbf, nu={NU:.2f}, gamma={GAMMA}",
    x=0.045,
    y=0.955,
    ha="left",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.045,
    0.918,
    "The grid search selected these settings. Panels 2–4 are "
    "the diagnostic: the flags track how well-connected the endpoints are, "
    "not whether the edge is correct.",
    ha="left",
    fontsize=10.5,
    color=INK2,
)

# 1 - PCA
ax = fig.add_subplot(gs[0, 0])
ax.scatter(
    P[~flag, 0], P[~flag, 1], s=3, c=BULK, lw=0, alpha=0.55, label=f"kept  n={(~flag).sum()}"
)
ax.scatter(
    P[flag, 0], P[flag, 1], s=5, c=ORANGE, lw=0, alpha=0.75, label=f"flagged  n={flag.sum()}"
)
ax.set_title("1 · Where the flags land — PCA of the standardised features")
ax.set_xlabel(f"PC1 ({100 * evr[0]:.0f}% var)")
ax.set_ylabel("PC2 (%.0f%% var)" % (100 * evr[1]))
leg = ax.legend(loc="upper right", frameon=False, markerscale=2.5)
for txt, col in zip(leg.get_texts(), [INK2, ORANGE], strict=False):
    txt.set_color(col)

# 2 - score vs degree
ax = fig.add_subplot(gs[0, 1])
ax.scatter(res.min_degree[~flag].clip(lower=1), res.score[~flag], s=3, c=BULK, lw=0, alpha=0.5)
ax.scatter(res.min_degree[flag].clip(lower=1), res.score[flag], s=4, c=ORANGE, lw=0, alpha=0.7)
ax.axhline(0, color=INK, lw=1, ls="--")
ax.text(1.1, 0.02, "decision boundary", fontsize=8.5, color=INK2)
ax.set_xscale("log")
ax.set_title(f"2 · The diagnostic: score vs node degree   (Spearman {rho:.2f})")
ax.set_xlabel("min(subject degree, object degree), log scale")
ax.set_ylabel("anomaly score  (negative = flagged)")

# 3 - flag rate by degree decile
ax = fig.add_subplot(gs[0, 2])
q = pd.qcut(res.min_degree.rank(method="first"), 10, labels=False)
rate = res.groupby(q).apply(lambda g: 100 * g.flag.mean())
mids = res.groupby(q).min_degree.median()
ax.bar(range(10), rate.values, color=ORANGE, width=0.72)
ax.set_xticks(range(10))
ax.set_xticklabels([f"{m:d}" for m in mids.values], fontsize=8)
ax.axhline(100 * flag.mean(), color=INK, lw=1, ls="--")
ax.text(
    9.4,
    100 * flag.mean() + 1.2,
    f"overall {100 * flag.mean():.0f}%",
    fontsize=8.5,
    color=INK2,
    ha="right",
)
ax.set_title("3 · Flag rate by degree decile")
ax.set_xlabel("median min-degree of the decile")
ax.set_ylabel("% of edges flagged")

# 4 - degree distributions
ax = fig.add_subplot(gs[1, 0])
bins = np.logspace(0, np.log10(max(res.min_degree.max(), 10)), 45)
ax.hist(
    res.min_degree[~flag].clip(lower=1),
    bins=bins,
    color=BULK,
    alpha=0.85,
    label=f"kept  (median {res.min_degree[~flag].median():.0f})",
)
ax.hist(
    res.min_degree[flag].clip(lower=1),
    bins=bins,
    color=ORANGE,
    alpha=0.8,
    label=f"flagged  (median {res.min_degree[flag].median():.0f})",
)
ax.set_xscale("log")
ax.set_title("4 · Who gets flagged: degree distribution")
ax.set_xlabel("min(subject degree, object degree), log scale")
ax.set_ylabel("edges")
leg = ax.legend(frameon=False)
for txt, col in zip(leg.get_texts(), [INK2, ORANGE], strict=False):
    txt.set_color(col)

# 5/6 - examples, spanning the two remaining cells
ax = fig.add_subplot(gs[1, 1:])
ax.axis("off")


def short(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def nm(c):
    v = name_of.get(c)
    return c if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)


ex = res.nsmallest(12, "score")
ax.set_title("5 · The twelve most anomalous edges by this model", pad=12)
cols = [0.0, 0.405, 0.795, 0.868, 0.935]
hdr = ["drug (subject)", "disease / object", "subj deg", "obj deg", "source"]
for cx, htxt, al in zip(cols, hdr, ["left", "left", "right", "right", "left"], strict=False):
    ax.text(
        cx + (0.05 if al == "right" else 0),
        0.962,
        htxt,
        fontsize=9,
        fontweight="bold",
        color=INK2,
        ha=al,
        transform=ax.transAxes,
    )
ax.plot([0, 1], [0.945, 0.945], color=GRID, lw=1, transform=ax.transAxes, clip_on=False)

PITCH = 0.0775
for i, (_, r) in enumerate(ex.iterrows()):
    y = 0.878 - i * PITCH
    if i % 2 == 0:
        ax.add_patch(
            plt.Rectangle(
                (0, y - 0.040),
                1,
                PITCH,
                transform=ax.transAxes,
                facecolor="#f0efec",
                edgecolor="none",
                zorder=0,
            )
        )
    # names on the first line, the CURIE beneath it
    ax.text(cols[0], y, short(nm(r.subject), 42), fontsize=8.6, color=INK, transform=ax.transAxes)
    ax.text(
        cols[0],
        y - 0.029,
        r.subject,
        fontsize=7.3,
        color=INK2,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.text(cols[1], y, short(nm(r.object), 46), fontsize=8.6, color=INK, transform=ax.transAxes)
    ax.text(
        cols[1],
        y - 0.029,
        r.object,
        fontsize=7.3,
        color=INK2,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.text(
        cols[2] + 0.05,
        y,
        f"{r.sdeg:.0f}",
        fontsize=8.6,
        color=ORANGE,
        ha="right",
        transform=ax.transAxes,
    )
    ax.text(
        cols[3] + 0.05,
        y,
        f"{r.odeg:.0f}",
        fontsize=8.6,
        color=ORANGE,
        ha="right",
        transform=ax.transAxes,
    )
    ax.text(
        cols[4],
        y,
        short(str(r.primary_knowledge_source).replace("infores:", ""), 14),
        fontsize=8.2,
        color=INK2,
        transform=ax.transAxes,
    )

stem = config.OUT / ("explain_rbf_{}".format(RUN.replace("original_", "")))
for ext, kw in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
    fig.savefig(str(stem) + ext, facecolor=SURFACE, bbox_inches="tight", **kw)
    print(f"wrote {stem}{ext}")

ex_cols = ["subject", "object", "sdeg", "odeg", "min_degree", "score", "primary_knowledge_source"]
top = res.nsmallest(50, "score")[ex_cols].copy()
top.insert(1, "subject_name", [nm(c) for c in top.subject])
top.insert(3, "object_name", [nm(c) for c in top.object])
top["subject_type"] = top.subject.map(type_of)
top["object_type"] = top.object.map(type_of)
top.to_csv(config.OUT / RUN / "top50_flagged_rbf.tsv", sep="\t", index=False)
print("wrote %s" % (config.OUT / RUN / "top50_flagged_rbf.tsv"))
