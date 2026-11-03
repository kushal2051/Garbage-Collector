"""Stage 3. One-class anomaly detection over the edges of a single predicate,
on RotatE embeddings, for garbage collection.

The method is the original oneclassSVM.py with GraphSAGE swapped for RotatE:
fit a OneClassSVM on concat(h, t) for every edge of one predicate, flag the ones
that fall outside the learned region. No labels and no split -- "anomalous"
means only "sparse region relative to the other edges of this predicate", which
is why stage 4 exists.

The output that matters is gc_candidates_<slug>.tsv: the flagged edges, ranked,
each carrying the context needed to triage it by hand --

  min_degree              a flag on a degree-1 node may mean "rare", not "wrong"
  object_type             a treats edge must point at a Disease or a
                          PhenotypicFeature; anything else is malformed by
                          definition, independent of any model
  type_rule_violation     that rule, evaluated
  rotate_dist_l2          RotatE's own score for the triple, for comparison

type_rule_violation is the model-independent column: where the detector and the
type rule agree, the edge is a defensible deletion without trusting the model.

Usage:  python detect.py [predicate] [nu]
        python detect.py biolink:treats 0.05
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
import rotate

PREDICATE = sys.argv[1] if len(sys.argv) > 1 else "biolink:treats"
NU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
TOP_N = 50  # size of the hand-read list
REFERENCE = "biolink:contraindicated_in"  # distance reference distribution
SLUG = PREDICATE.split(":")[1]

config.require(config.EDGES_TSV, config.DEGREES_TSV, config.NODE_TYPES)
config.refuse_overwrite(
    config.OUT / (f"gc_candidates_{SLUG}.tsv"), config.OUT / (f"ranked_{SLUG}.tsv")
)

emb = rotate.Embeddings()
rid = emb.relation_id(PREDICATE)
edges = pd.read_csv(config.EDGES_TSV, sep="\t")
e, dropped = rotate.map_pairs(edges, PREDICATE, emb.row_of)

print(f"predicate            {PREDICATE} (relation_id {rid})")
print(f"edges                {len(e)}  ({dropped} dropped, endpoint not in embeddings)")

X = rotate.concat_features(emb, e)
e["rotate_dist_l2"], e["rotate_dist_l1"] = rotate.rotate_distance(emb, e, rid)
e, _, type_of = rotate.context(e)

# ------------------------------------------------------------------ detect
Xs = StandardScaler().fit_transform(X)
model = OneClassSVM(kernel="rbf", nu=NU, gamma="scale").fit(Xs)
e["score"] = model.decision_function(Xs)
e["flagged"] = e["score"] < 0

e["type_rule_violation"] = ~e.object_type.isin(config.VALID_TREATS_OBJECT_TYPES)

flagged = e.flagged
summary = pd.DataFrame(
    [
        {
            "predicate": PREDICATE,
            "nu": NU,
            "features": "concat",
            "dim": X.shape[1],
            "edges": len(e),
            "flagged": int(flagged.sum()),
            "flag_rate": round(float(flagged.mean()), 4),
            # is the score just reading node degree back to us?
            "spearman_score_vs_min_degree": round(
                float(spearmanr(e.score, e.min_degree).statistic), 3
            ),
            "median_degree_flagged": float(e.loc[flagged, "min_degree"].median()),
            "median_degree_unflagged": float(e.loc[~flagged, "min_degree"].median()),
            "spearman_score_vs_rotate_dist": round(
                float(spearmanr(e.score, e.rotate_dist_l2).statistic), 3
            ),
            "type_rule_violation_rate_flagged": round(
                float(e.loc[flagged, "type_rule_violation"].mean()), 4
            ),
            "type_rule_violation_rate_unflagged": round(
                float(e.loc[~flagged, "type_rule_violation"].mean()), 4
            ),
        }
    ]
)

print("\n=== detector summary ===")
print(summary.to_string(index=False))

print("\n=== RotatE's own distance, for reference ===")
for col in ("rotate_dist_l2", "rotate_dist_l1"):
    v = e[col]
    print(f"{col:<16} median {v.median():9.2f}   p95 {v.quantile(0.95):9.2f}")

# ------------------------------------------------------------------ outputs
COLS = [
    "subject",
    "subject_type",
    "subject_degree",
    "object",
    "object_type",
    "object_degree",
    "min_degree",
    "type_rule_violation",
    "primary_knowledge_source",
    "score",
    "flagged",
    "rotate_dist_l2",
    "rotate_dist_l1",
]

ranked = e.sort_values("score")[COLS]
ranked.to_csv(config.OUT / (f"ranked_{SLUG}.tsv"), sep="\t", index=False)
ranked[ranked.flagged].to_csv(config.OUT / (f"gc_candidates_{SLUG}.tsv"), sep="\t", index=False)
ranked.head(TOP_N).to_csv(config.OUT / f"top{TOP_N}_{SLUG}.tsv", sep="\t", index=False)
summary.to_csv(config.OUT / (f"summary_{SLUG}.tsv"), sep="\t", index=False)

# 2-D projections and a reference distribution for the explanatory figure.
# PCA is a viewing device only -- the SVM is fit in the full space above, so
# these coordinates explain the result rather than producing it.
viz = {}
p = PCA(n_components=2).fit(Xs)
viz["pca_concat"] = p.transform(Xs)
viz["evr_concat"] = p.explained_variance_ratio_

# a sample of the raw node space, to show what the embedding itself looks like
samp = np.sort(np.random.default_rng(0).choice(emb.nodes.shape[0], size=25000, replace=False))
node_sample = emb.vectors(samp)
pn = PCA(n_components=2).fit(node_sample)
viz["pca_nodes"] = pn.transform(node_sample)
viz["evr_nodes"] = pn.explained_variance_ratio_
viz["node_sample_types"] = np.array(
    [type_of.get(c, "unknown") for c in emb.curies[samp]], dtype=object
)

# the same RotatE distance for the predicate used as negatives in stage 4:
# overlap here is the reason the detector has a ceiling
if REFERENCE != PREDICATE and REFERENCE in emb.rel_of:
    r, _ = rotate.map_pairs(edges, REFERENCE, emb.row_of)
    viz["ref_dist_l2"] = rotate.rotate_distance(emb, r, emb.relation_id(REFERENCE))[0]
    viz["ref_label"] = REFERENCE

np.savez_compressed(
    config.OUT / (f"viz_{SLUG}.npz"),
    score=e.score.values,
    flagged=e.flagged.values,
    min_degree=e.min_degree.values.astype(float),
    dist_l2=e.rotate_dist_l2.values,
    type_rule_violation=e.type_rule_violation.values,
    predicate=PREDICATE,
    nu=NU,
    **viz,
)

print(f"\nwrote {config.OUT / f'gc_candidates_{SLUG}.tsv'}")
print(f"      {config.OUT / f'ranked_{SLUG}.tsv'}")
print(f"      {config.OUT / f'top{TOP_N}_{SLUG}.tsv'}")
print(f"      {config.OUT / f'viz_{SLUG}.npz'}")

print("\n=== top 15 garbage-collection candidates ===")
show = [
    "subject",
    "object",
    "object_type",
    "min_degree",
    "type_rule_violation",
    "score",
    "rotate_dist_l2",
]
print(ranked.head(15)[show].to_string(index=False))

n_both = int((flagged & e.type_rule_violation).sum())
print(
    f"\nflagged AND violating the type rule: {n_both} edges (deletable without trusting the model)"
)
