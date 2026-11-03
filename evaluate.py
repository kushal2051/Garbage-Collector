"""Stage 4. Does the detector actually separate good treats edges from bad ones?

Stage 3 can only say "sparse region". To know whether sparse means wrong, the
scores need something known-bad to rank against. The construction:

  positives   biolink:treats edges, split 60/20/20. The SVM sees only the 60%
              fit split -- and only positives, as a one-class method must.
  negatives   biolink:contraindicated_in edges. Never fitted on. A chemical
              contraindicated in a condition is the cleanest available stand-in
              for a treats edge that should not exist: same node types, same
              direction, opposite clinical meaning.

Two things the earlier version of this script got wrong and this one does not:

  1. No sampled test draws. Every held-out positive is scored against EVERY
     negative in the test pool. ROC AUC is invariant to class ratio, so this is
     exact rather than an average over draws. PR AUC is not invariant, so it is
     reported next to its prevalence baseline -- at prevalence 0.87 a PR AUC of
     0.94 is worth almost nothing, and reading it as success is the trap.
  2. The variance reported is across 10 independent splits of the positives.
     Holding the split fixed and resampling negatives, as before, measured
     nothing (+-0.003).

nu is re-selected on each seed's own validation split, so no information from a
seed's test set reaches its model.

Reference rows, reported alongside and NOT fitted:
  distance    ||h o r - t||, RotatE's own score. One number, no training.
  degree      log1p of both endpoint degrees. The null hypothesis: that any
              apparent skill is just "rare nodes are suspicious".
Without these two rows the concat number is uninterpretable.

Usage:  python evaluate.py [--seeds 10] [--out-prefix evaluation]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
import rotate

SVM_MODELS = ("concat",)
BASELINES = ("distance", "degree")

ap = argparse.ArgumentParser()
ap.add_argument("--positive", default="biolink:treats")
ap.add_argument("--negative", default="biolink:contraindicated_in")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--nu-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
ap.add_argument("--no-baselines", action="store_true",
                help="report the SVM alone (the concat number is then not "
                     "interpretable on its own -- see the docstring)")
ap.add_argument("--balanced-val", action="store_true",
                help="subsample validation negatives to one per validation "
                     "positive (nu is then chosen on a 1:1 set)")
ap.add_argument("--balanced-test", action="store_true",
                help="subsample test negatives to one per test positive")
ap.add_argument("--out-prefix", default="evaluation")
args = ap.parse_args()

NU_GRID = [float(x) for x in args.nu_grid.split(",")]
MODELS = list(SVM_MODELS) + ([] if args.no_baselines else list(BASELINES))
KIND = {m: ("svm" if m in SVM_MODELS else "baseline") for m in MODELS}

# fail in the first second rather than after an hour of compute
config.require(config.EDGES_TSV, config.DEGREES_TSV)
config.refuse_overwrite(*[config.OUT / (args.out_prefix + s) for s in
                          ("_per_seed.tsv", "_summary.tsv",
                           "_seen_unseen.tsv", "_meta.json")])

emb = rotate.Embeddings()
RID = emb.relation_id(args.positive)
edges = pd.read_csv(config.EDGES_TSV, sep="\t")

deg = pd.read_csv(config.DEGREES_TSV, sep="\t")
degree = dict(zip(deg.curie, deg.in_degree + deg.out_degree))

pos, _ = rotate.map_pairs(edges, args.positive, emb.row_of)
neg, _ = rotate.map_pairs(edges, args.negative, emb.row_of)


def features(e):
    """The one feature set the SVM is fit on, plus the two reference columns.

    Note the rotation used is always the POSITIVE predicate's: the question is
    how well a contraindicated_in pair satisfies the treats relation, not its
    own.
    """
    dist = rotate.rotate_distance(emb, e, RID)[0]
    dg = np.column_stack([
        np.log1p([degree.get(x, 0) for x in e.subject]),
        np.log1p([degree.get(x, 0) for x in e.object])])
    return {"concat": rotate.concat_features(emb, e),
            "distance": dist.reshape(-1, 1),
            "degree": dg}


F_pos, F_neg = features(pos), features(neg)
print("positives %s %d   negatives %s %d"
      % (args.positive, len(pos), args.negative, len(neg)), flush=True)


def metrics(y, s):
    pr, rc, _ = precision_recall_curve(y, s)
    return roc_auc_score(y, s), auc(rc, pr)


def score(name, nu, fit_i, pos_i, neg_i):
    """Anomaly scores for all of pos_i then all of neg_i (higher = anomalous)."""
    if KIND[name] == "baseline":
        X = np.vstack([F_pos[name][pos_i], F_neg[name][neg_i]])
        return X[:, 0] if X.shape[1] == 1 else -X.sum(1)
    sc = StandardScaler().fit(F_pos[name][fit_i])
    m = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(
        sc.transform(F_pos[name][fit_i]))
    X = np.vstack([sc.transform(F_pos[name][pos_i]),
                   sc.transform(F_neg[name][neg_i])])
    return -m.decision_function(X)


records, chosen_nu, seen_rows = [], {m: [] for m in MODELS}, []

for seed in range(args.seeds):
    rng = np.random.default_rng(1000 + seed)
    p = rng.permutation(len(pos))
    n_fit, n_val = int(0.6 * len(pos)), int(0.2 * len(pos))
    fit_i, val_i, test_i = p[:n_fit], p[n_fit:n_fit + n_val], p[n_fit + n_val:]

    q = rng.permutation(len(neg))
    neg_val, neg_test = q[:len(q) // 2], q[len(q) // 2:]
    if args.balanced_val:                       # one negative per positive
        neg_val = neg_val[:len(val_i)]
    if args.balanced_test:
        neg_test = neg_test[:len(test_i)]

    y_val = np.concatenate([np.zeros(len(val_i)), np.ones(len(neg_val))])
    y_test = np.concatenate([np.zeros(len(test_i)), np.ones(len(neg_test))])

    # a held-out positive whose endpoints the SVM never saw is a harder case
    seen = set(pos.subject.iloc[fit_i]) | set(pos.object.iloc[fit_i])
    both_seen = np.array([(a in seen) and (b in seen) for a, b in
                          zip(pos.subject.iloc[test_i], pos.object.iloc[test_i])])

    for name in MODELS:
        if KIND[name] == "baseline":
            nu = None
        else:                                   # nu chosen on this seed's validation
            nu = max((roc_auc_score(y_val, score(name, g, fit_i, val_i, neg_val)), g)
                     for g in NU_GRID)[1]
        chosen_nu[name].append(nu)

        s = score(name, nu, fit_i, test_i, neg_test)
        roc, prc = metrics(y_test, s)
        records.append({"seed": seed, "model": name, "kind": KIND[name],
                        "nu": nu, "roc_auc": roc, "pr_auc": prc})

        s_pos, s_neg = s[:len(test_i)], s[len(test_i):]
        for label, mask in (("both_seen", both_seen), ("has_unseen", ~both_seen)):
            if mask.sum() >= 20:
                yy = np.concatenate([np.zeros(int(mask.sum())), np.ones(len(s_neg))])
                seen_rows.append({"seed": seed, "model": name, "subset": label,
                                  "n_pos": int(mask.sum()),
                                  "roc_auc": roc_auc_score(
                                      yy, np.concatenate([s_pos[mask], s_neg]))})
    print("seed %d done" % seed, flush=True)

df = pd.DataFrame(records)
summary = (df.groupby(["model", "kind"])
             .agg(roc_mean=("roc_auc", "mean"), roc_std=("roc_auc", "std"),
                  roc_min=("roc_auc", "min"), roc_max=("roc_auc", "max"),
                  pr_mean=("pr_auc", "mean"), pr_std=("pr_auc", "std"))
             .sort_values("roc_mean", ascending=False).reset_index())

n_test_pos = len(pos) - int(0.6 * len(pos)) - int(0.2 * len(pos))
n_val_pos = int(0.2 * len(pos))
n_val_neg = n_val_pos if args.balanced_val else len(neg) // 2
n_test_neg = n_test_pos if args.balanced_test else len(neg) - len(neg) // 2
prevalence = n_test_neg / (n_test_neg + n_test_pos)

print("\n=== %d independent 60/20/20 splits ===" % args.seeds)
print("fit        %d positives, 0 negatives" % int(0.6 * len(pos)))
print("validation %d positives + %d negatives   (nu selection only)"
      % (n_val_pos, n_val_neg))
print("test       %d positives + %d negatives   prevalence %.3f"
      % (n_test_pos, n_test_neg, prevalence))
print(summary.to_string(index=False, float_format=lambda v: "%.3f" % v))
print("\nROC AUC 0.500 = chance.")
print("PR AUC baseline (prevalence of the anomaly class) = %.3f -- any PR AUC "
      "near this is chance, not skill." % prevalence)
print("\nnu chosen per seed:")
for m in MODELS:
    if KIND[m] == "svm":
        print("  %-9s %s" % (m, chosen_nu[m]))

sdf = pd.DataFrame(seen_rows)
print("\n=== held-out positives split by whether both endpoints were in the fit set ===")
print(sdf.groupby(["model", "subset"]).roc_auc.agg(["mean", "std"])
      .reset_index().to_string(index=False, float_format=lambda v: "%.3f" % v))

pre = args.out_prefix
df.to_csv(config.OUT / ("%s_per_seed.tsv" % pre), sep="\t", index=False)
summary.to_csv(config.OUT / ("%s_summary.tsv" % pre), sep="\t", index=False)
sdf.to_csv(config.OUT / ("%s_seen_unseen.tsv" % pre), sep="\t", index=False)
with open(config.OUT / ("%s_meta.json" % pre), "w") as fh:
    json.dump({"positive": args.positive, "negative": args.negative,
               "seeds": args.seeds, "prevalence": prevalence,
               "models": MODELS, "kind": KIND,
               "balanced_val": args.balanced_val,
               "balanced_test": args.balanced_test,
               "n_val_pos": n_val_pos, "n_val_neg": n_val_neg,
               "n_test_pos": n_test_pos, "n_test_neg": n_test_neg,
               "chosen_nu": chosen_nu,
               "note": "no negative sampling; variance is across positive splits"},
              fh, indent=2, default=str)
print("\nwrote %s" % (config.OUT / ("%s_summary.tsv" % pre)))
