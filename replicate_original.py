"""Faithful port of SVM/oneclassSVM.py, with RotatE node embeddings in place of
the GraphSAGE ones and the robokop_30fd_filtered graph in place of the Neo4j
edge export. Everything else follows the original step for step, including the
parts I think are wrong -- the point is to see what the original method gives on
these embeddings before changing anything.

Steps, matching the original:
  1. pair features = concat(subject_emb, object_emb)
  2. UMAP of the raw features
  3. L2 row-normalize            (normalize_embeddings, original L46-48)
  4. UMAP of the normalized features
  5. StandardScaler
  6. UMAP of the standardized features
  7. 10% random subsample
  8. grid search 4 nu x 4 gamma x 2 kernels = 32, each fit on the subsample,
     selected by max mean decision_function on that same subsample
     (original L166-192 -- kept as written)
  9. refit the winner on the full standardized set, predict
 10. UMAP of the results, standardized and raw
 11. write the per-edge predictions CSV

Two deliberate departures, neither touching the analysis:
  - mlflow logging dropped; the grid is written to a CSV instead
  - the subsample draw is seeded, so the run reproduces

Usage:  python replicate_original.py [predicate]
"""
import sys
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import argparse

import config

EMB = config.EMB

_ap = argparse.ArgumentParser()
_ap.add_argument("predicate", nargs="?", default="biolink:treats")
_ap.add_argument("--kernels", default="rbf,poly",
                 help="comma-separated subset of the original grid's kernels")
_ap.add_argument("--gammas", default="scale,auto,0.01,0.1")
_ap.add_argument("--nus", default="0.01,0.05,0.1,0.2")
_ap.add_argument("--tag", default="",
                 help="suffix for the output directory, so a restricted grid "
                      "does not overwrite the faithful replication")
_args = _ap.parse_args()

PREDICATE = _args.predicate
SLUG = PREDICATE.split(":")[1]
OUT = config.OUT / ("original_%s%s" % (SLUG, _args.tag))
if OUT.exists():
    raise SystemExit("refusing to overwrite %s -- pass a different --tag" % OUT)
OUT.mkdir(parents=True, exist_ok=True)


# ---- original helpers, carried over -------------------------------------
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def visualize_embeddings(embeddings, title, output_file):
    umap_2d = umap.UMAP(n_components=2, n_jobs=-1, random_state=42)
    reduced = umap_2d.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], s=1, c="red", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True)
    plt.savefig(output_file)
    plt.close()
    return reduced


# ---- inputs --------------------------------------------------------------
nodes = np.load(EMB / "node_embeddings.npy", mmap_mode="r")
idx = pd.read_csv(EMB / "node_index.tsv", sep="\t", usecols=["node_id", "curie"])
row_of = dict(zip(idx.curie, idx.node_id))

config.require(config.EDGES_TSV)
edges = pd.read_csv(config.EDGES_TSV, sep="\t")
e = edges[edges.predicate == PREDICATE].copy()
e["s_row"] = e.subject.map(row_of)
e["o_row"] = e.object.map(row_of)
e = e.dropna(subset=["s_row", "o_row"]).reset_index(drop=True)
print("Embeddings for ROBOKOP-KG (RotatE, 512 real / 256 complex dims).")
print("%s edges: %d" % (PREDICATE, len(e)))

h = np.asarray(nodes[e.s_row.values.astype(int)], dtype=np.float64)
t = np.asarray(nodes[e.o_row.values.astype(int)], dtype=np.float64)
training_np = np.hstack([h, t])
print("Number of pairs in training set: %d" % training_np.shape[0])
print("Feature matrix: %s" % (training_np.shape,))

# ---- steps 2-6 -----------------------------------------------------------
umap_raw = visualize_embeddings(
    training_np, "UMAP 2D Projection of Embeddings", OUT / "UMAP_2D_Projection.pdf")

normalized_training = normalize_embeddings(training_np)
norms = np.linalg.norm(normalized_training, axis=1)
print("Are embeddings normalized? %s"
      % np.allclose(norms, np.ones_like(norms), atol=1e-5))

visualize_embeddings(normalized_training, "UMAP 2D Projection of Normalized Embeddings",
                     OUT / "UMAP_2D_Projection_Normalized.pdf")

scaler = StandardScaler()
standardized_embeddings = scaler.fit_transform(normalized_training)

sample_fraction = 0.1
sample_size = int(sample_fraction * len(standardized_embeddings))
random_indices = np.random.default_rng(42).choice(
    len(standardized_embeddings), size=sample_size, replace=False)
sampled_data = standardized_embeddings[random_indices]

umap_std = visualize_embeddings(
    standardized_embeddings, "UMAP 2D Projection of Standarized Embeddings",
    OUT / "UMAP_2D_Projection_Standarized.pdf")

# ---- step 8: the grid search, exactly as the original wrote it -----------
param_grid = {
    "nu": [float(x) for x in _args.nus.split(",")],
    "gamma": [g if g in ("scale", "auto") else float(g)
              for g in _args.gammas.split(",")],
    "kernel": _args.kernels.split(","),
}
print("grid:", param_grid)
param_dicts = [dict(zip(param_grid.keys(), v))
               for v in product(*param_grid.values())]

best_params, best_score, grid_rows = None, -np.inf, []
for params in param_dicts:
    print("Running Manual Grid Search with params:", params)
    model = OneClassSVM(kernel=params["kernel"], nu=params["nu"], gamma=params["gamma"])
    model.fit(sampled_data)
    score = model.decision_function(sampled_data).mean()
    grid_rows.append({**params,
                      "mean_decision_function_score": float(score),
                      "frac_flagged_on_sample":
                          float((model.decision_function(sampled_data) < 0).mean())})
    if score > best_score:
        best_score = score
        best_params = {"kernel": params["kernel"], "nu": params["nu"],
                       "gamma": params["gamma"]}

pd.DataFrame(grid_rows).sort_values(
    "mean_decision_function_score", ascending=False).to_csv(
    OUT / "grid_search.csv", index=False)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# ---- step 9: refit and predict ------------------------------------------
oc_svm = OneClassSVM(**best_params)
oc_svm.fit(standardized_embeddings)
predictions = oc_svm.predict(standardized_embeddings)
print("Flagged as anomaly: %d / %d (%.1f%%)"
      % ((predictions == -1).sum(), len(predictions),
         100 * (predictions == -1).mean()))

# ---- step 10: result plots ----------------------------------------------
for reduced, title, fname in (
        (umap_std, "UMAP Visualization of One-Class SVM Results on Standardized Embeddings",
         "UMAP_2D_Projection_Results_Standardized.pdf"),
        (umap_raw, "UMAP Visualization of One-Class SVM Results",
         "UMAP_2D_Projection_Results.pdf")):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(reduced[predictions == 1][:, 0], reduced[predictions == 1][:, 1],
                label="Normal", alpha=0.7, s=1)
    plt.scatter(reduced[predictions == -1][:, 0], reduced[predictions == -1][:, 1],
                label="Anomaly", alpha=0.7, s=2, color="red")
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    ax.grid(True)
    plt.legend()
    plt.savefig(OUT / fname)
    plt.close()

# ---- step 11: results ----------------------------------------------------
e["predictions"] = predictions
e.drop(columns=["s_row", "o_row"]).to_csv(
    OUT / ("OneClassSVM_Results_%s.csv" % SLUG), index=False)
print("\nwrote %s" % OUT)
