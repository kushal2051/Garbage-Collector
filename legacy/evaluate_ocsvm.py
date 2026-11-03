# ---------------------------------------------------------------------------
# SUPERSEDED BY ../evaluate.py -- kept as an independent cross-check, not as a
# live entry point.
#
# This is the FIRST implementation of the treats vs contraindicated_in
# comparison, and the origin of that design: treats = normal class,
# contraindicated_in = type-matched hard negatives, every pair scored through
# the treats rotation. evaluate.py keeps that design unchanged.
#
# What evaluate.py fixes is the protocol, not the question:
#   - here, each test draw is 1:1 (1,932 positives vs 1,932 sampled negatives)
#     and the error bar comes from resampling negatives. ROC AUC is invariant
#     to class ratio, so the sampling was unnecessary, and the resulting
#     +-0.003 measured sampling noise rather than anything about the method.
#   - evaluate.py scores every held-out positive against every negative
#     (exact), and takes its error bar across 10 independent splits of the
#     positives.
#
# The two agree within 0.01 on all four models -- see
# ../results/cross_check/evaluation_auc.tsv next to ../results/evaluation_summary.tsv.
# That agreement is the reason this file is still here. It was run once, as
# Slurm job 7261544 (2026-10-29), and its outputs are in results/cross_check/.
#
# It still contains the `residual` feature set, which the current pipeline
# drops; that is expected and is part of what it is a record of.
# ---------------------------------------------------------------------------
"""Supervised evaluation of one-class detectors, in the style of
hiram64/ocsvm-anomaly-detection: fit on the normal class only, test on held-out
normals plus a labelled anomaly class, and report ROC AUC / PR AUC.

The mapping to this graph:
    normal class  = biolink:treats edges
    anomaly class = biolink:contraindicated_in edges (type-matched hard negatives)

Every candidate pair is scored as if it were a `treats` edge -- the residual and
distance features always use the `treats` rotation -- because the question is
"does this pair look like a treats edge?"

Four models on identical splits, so the comparison is apples to apples:
    concat     OneClassSVM on concat(h, t)
    residual   OneClassSVM on h o r - t
    distance   RotatE distance ||h o r - t||_2, no SVM at all
    degree     OneClassSVM on [log1p(deg_h), log1p(deg_t)] -- the control

Protocol (stricter than the reference repo, which selects nu on its test set):
    positives split 60 / 20 / 20 into fit / validation / test
    negatives split disjointly into validation and test pools
    nu chosen per model on the validation AUC, then reported on the test draws
    10 test draws, averaged

Usage:  python evaluate_ocsvm.py [--anomaly-ratio 1.0] [--draws 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

HERE = Path(__file__).resolve().parent
EMB = Path("/work/users/k/k/kkoirala/nvidia-hackathon/rotate_original_embeddings")

ap = argparse.ArgumentParser()
ap.add_argument("--positive", default="biolink:treats")
ap.add_argument("--negative", default="biolink:contraindicated_in")
ap.add_argument("--anomaly-ratio", type=float, default=1.0,
                help="anomalies per held-out normal in each test draw")
ap.add_argument("--draws", type=int, default=10)
ap.add_argument("--nu-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
args = ap.parse_args()
NU_GRID = [float(x) for x in args.nu_grid.split(",")]
SEED = 42

# ---------------------------------------------------------------- inputs
nodes = np.load(EMB / "node_embeddings.npy", mmap_mode="r")
rots = np.load(EMB / "relation_rotations.npy")
D = nodes.shape[1] // 2

idx = pd.read_csv(EMB / "node_index.tsv", sep="\t", usecols=["node_id", "curie"])
row_of = dict(zip(idx.curie, idx.node_id))
rel = pd.read_csv(EMB / "relation_index.tsv", sep="\t")
rel_of = dict(zip(rel.predicate, rel.relation_id))
RID = rel_of[args.positive]                       # always the positive relation

deg = pd.read_csv(HERE / "out/degrees.tsv", sep="\t")
degree = dict(zip(deg.curie, deg.in_degree + deg.out_degree))

edges = pd.read_csv(HERE / "out/edges.tsv", sep="\t")


def pairs(pred):
    e = edges[edges.predicate == pred].copy()
    e["s_row"] = e.subject.map(row_of)
    e["o_row"] = e.object.map(row_of)
    return e.dropna(subset=["s_row", "o_row"]).reset_index(drop=True)


pos, neg = pairs(args.positive), pairs(args.negative)
print("positives (normal)  %s  %d" % (args.positive, len(pos)))
print("negatives (anomaly) %s  %d" % (args.negative, len(neg)))


def features(e):
    """All four feature sets for a set of pairs, under the positive relation."""
    h = np.asarray(nodes[e.s_row.values.astype(int)], dtype=np.float64)
    t = np.asarray(nodes[e.o_row.values.astype(int)], dtype=np.float64)
    c, s = rots[RID, :D].astype(np.float64), rots[RID, D:].astype(np.float64)
    zre = h[:, :D] * c - h[:, D:] * s - t[:, :D]
    zim = h[:, :D] * s + h[:, D:] * c - t[:, D:]
    dist = np.sqrt((zre ** 2 + zim ** 2).sum(1))
    dg = np.column_stack([
        np.log1p([degree.get(x, 0) for x in e.subject]),
        np.log1p([degree.get(x, 0) for x in e.object])])
    return {"concat": np.hstack([h, t]),
            "residual": np.hstack([zre, zim]),
            "distance": dist.reshape(-1, 1),
            "degree": dg}


F_pos, F_neg = features(pos), features(neg)

# ---------------------------------------------------------------- splits
rng = np.random.default_rng(SEED)
p = rng.permutation(len(pos))
n_fit, n_val = int(0.6 * len(pos)), int(0.2 * len(pos))
fit_i, val_i, test_i = p[:n_fit], p[n_fit:n_fit + n_val], p[n_fit + n_val:]

q = rng.permutation(len(neg))
neg_val_pool, neg_test_pool = q[:len(q) // 2], q[len(q) // 2:]
print("positives  fit %d  val %d  test %d" % (len(fit_i), len(val_i), len(test_i)))
print("negative pools  val %d  test %d" % (len(neg_val_pool), len(neg_test_pool)))

# nodes the model actually saw during fitting, for the seen/unseen breakdown
seen_nodes = set(pos.subject.iloc[fit_i]) | set(pos.object.iloc[fit_i])

MODELS = ["concat", "residual", "distance", "degree"]


def scores(name, nu, pos_i, neg_i):
    """Anomaly scores (higher = more anomalous) for a positive/negative draw."""
    if name == "distance":
        # no model: the RotatE distance is already an anomaly score
        return np.concatenate([F_pos[name][pos_i, 0], F_neg[name][neg_i, 0]])
    sc = StandardScaler().fit(F_pos[name][fit_i])
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    model.fit(sc.transform(F_pos[name][fit_i]))
    X = np.vstack([sc.transform(F_pos[name][pos_i]),
                   sc.transform(F_neg[name][neg_i])])
    return -model.decision_function(X)


def evaluate(y, s):
    pr, rc, _ = precision_recall_curve(y, s)
    return roc_auc_score(y, s), auc(rc, pr)


def draw(seed, pos_i, neg_pool):
    r = np.random.default_rng(seed)
    k = min(int(args.anomaly_ratio * len(pos_i)), len(neg_pool))
    neg_i = r.choice(neg_pool, size=k, replace=False)
    y = np.concatenate([np.zeros(len(pos_i)), np.ones(k)])
    return neg_i, y


# ---------------------------------------------------------------- validation
best_nu = {}
for name in MODELS:
    if name == "distance":
        best_nu[name] = None
        continue
    neg_i, y = draw(SEED, val_i, neg_val_pool)
    aucs = [(evaluate(y, scores(name, nu, val_i, neg_i))[0], nu) for nu in NU_GRID]
    best_nu[name] = max(aucs)[1]
    print("%-9s validation: best nu = %.1f (ROC AUC %.3f)"
          % (name, max(aucs)[1], max(aucs)[0]))

# ---------------------------------------------------------------- test
rows, per_draw = [], {}
for name in MODELS:
    roc, prc = [], []
    for d in range(args.draws):
        neg_i, y = draw(1000 + d, test_i, neg_test_pool)
        s = scores(name, best_nu[name], test_i, neg_i)
        a, b = evaluate(y, s)
        roc.append(a)
        prc.append(b)
    per_draw[name] = roc
    rows.append({"model": name, "nu": best_nu[name],
                 "roc_auc_mean": np.mean(roc), "roc_auc_std": np.std(roc),
                 "pr_auc_mean": np.mean(prc), "pr_auc_std": np.std(prc)})

res = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
print("\n=== test set: %d draws, %d normals + %d anomalies each ==="
      % (args.draws, len(test_i),
         min(int(args.anomaly_ratio * len(test_i)), len(neg_test_pool))))
print(res.to_string(index=False, float_format=lambda v: "%.3f" % v))
print("\nPR AUC baseline (prevalence) = %.3f"
      % (args.anomaly_ratio / (1 + args.anomaly_ratio)))

# ---------------------------------------------------------------- seen/unseen
print("\n=== held-out positives split by whether endpoints were in the fit set ===")
both_seen = np.array([(s in seen_nodes) and (o in seen_nodes)
                      for s, o in zip(pos.subject.iloc[test_i],
                                      pos.object.iloc[test_i])])
print("both endpoints seen %d   at least one unseen %d"
      % (both_seen.sum(), (~both_seen).sum()))
sub_rows = []
for name in MODELS:
    neg_i, _ = draw(1000, test_i, neg_test_pool)
    s = scores(name, best_nu[name], test_i, neg_i)
    s_pos, s_neg = s[:len(test_i)], s[len(test_i):]
    row = {"model": name}
    for label, mask in (("both_seen", both_seen), ("has_unseen", ~both_seen)):
        if mask.sum() < 20:
            row[label] = float("nan")
            continue
        y = np.concatenate([np.zeros(int(mask.sum())), np.ones(len(s_neg))])
        row[label] = roc_auc_score(y, np.concatenate([s_pos[mask], s_neg]))
    sub_rows.append(row)
print(pd.DataFrame(sub_rows).to_string(index=False,
                                       float_format=lambda v: "%.3f" % v))

out = HERE / "out"
res.to_csv(out / "evaluation_auc.tsv", sep="\t", index=False)
pd.DataFrame(sub_rows).to_csv(out / "evaluation_seen_unseen.tsv", sep="\t", index=False)
with open(out / "evaluation_meta.json", "w") as fh:
    json.dump({"positive": args.positive, "negative": args.negative,
               "anomaly_ratio": args.anomaly_ratio, "draws": args.draws,
               "best_nu": best_nu, "n_fit": len(fit_i), "n_test": len(test_i),
               "roc_per_draw": {k: list(map(float, v)) for k, v in per_draw.items()}},
              fh, indent=2)
print("\nwrote %s" % (out / "evaluation_auc.tsv"))
