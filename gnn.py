"""Stage 5c. GraphSAGE link prediction on a prepared graph, with the baseline and the spread.

Two ways to run it:

    --split out/split_<tag>_pairs.tsv    the shared split, for the Pharos comparison. Both
                                         graphs are scored on identical test rows and identical
                                         negatives, which is the only way the two numbers can
                                         be put side by side.
    no --split                           an own-graph split, for a single graph's own number.

Every number is printed beside the two things without which it cannot be read, as AGENTS.md
requires:

    degree          a degree-only baseline. "Our graph beats Pharos" must not reduce to "our
                    graph puts more edges on hub nodes" -- the failure mode that invalidated
                    the node2vec run on the mf1 graph.
    both_seen / has_unseen
                    test edges split by whether both endpoints appear in the training edges.
                    RotatE fell from 0.881 to 0.685 across that line; a model that only
                    memorises node identity collapses on the unseen half.

Usage:
    python gnn.py --graph kg     --predicate biolink:genetically_associated_with \
                  --split out/split_genetically_associated_with_pairs.tsv --tag kg_gaw
    python gnn.py --graph pharos --predicate biolink:genetically_associated_with \
                  --split out/split_genetically_associated_with_pairs.tsv --tag pharos_gaw
"""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import config
import kgraph
import sage

MIN_SUBSET = 20  # a seen/unseen subset smaller than this is not reported

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="kg", help="a tag prepared by kg_prepare.py, or a .npz path")
ap.add_argument("--predicate", default="biolink:genetically_associated_with")
ap.add_argument("--split", default=None, help="a pairs.tsv from pharos_split.py")
ap.add_argument("--seeds", type=int, default=10, help="independent splits")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=5, help="epochs without a validation gain")
ap.add_argument("--batch", type=int, default=4096, help="positive edges per step")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument(
    "--embed-dim",
    type=int,
    default=64,
    help="learned vector per node, concatenated to its features; 0 = features only, fully "
    "inductive and much weaker (see sage.LinkPredictor)",
)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=0.005)
ap.add_argument("--fanouts", default="10,5", help="neighbours per hop while training")
ap.add_argument("--eval-fanouts", default="25,10", help="wider at inference")
ap.add_argument("--train-frac", type=float, default=0.8)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--tag", default="gnn", help="output prefix")
ap.add_argument("--no-write", action="store_true")
args = ap.parse_args()

FANOUTS = tuple(int(x) for x in args.fanouts.split(","))
EVAL_FANOUTS = tuple(int(x) for x in args.eval_fanouts.split(","))
DEVICE = torch.device(args.device)

npz = config.OUT / f"{args.graph}.npz" if not args.graph.endswith(".npz") else args.graph
config.require(npz)
outputs = [
    config.OUT / f"{args.tag}{suffix}"
    for suffix in ("_per_seed.tsv", "_summary.tsv", "_seen_unseen.tsv", "_meta.json")
]
if not args.no_write:
    config.refuse_overwrite(*outputs)

graph = kgraph.load(npz)
print(graph.describe(), flush=True)
TARGET = graph.predicate_id(args.predicate)


def metrics(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    """ROC AUC first; PR AUC only ever next to the prevalence it is read against."""
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(auc(recall, precision)),
        "prevalence": float(len(positive) / (len(positive) + len(negative))),
        "n_pos": int(len(positive)),
        "n_neg": int(len(negative)),
    }


# ------------------------------------------------------------------ the split and the negatives

index = graph.index()


def load_shared_split(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The shared pairs, as node ids, with every pair either graph holds as the filter set.

    Negatives are derived from this file alone -- its node pools, its pair set, the seed -- and
    never from the graph being scored, so both runs draw the SAME negatives. Filtered against
    every pair either graph asserts, so a negative is not an edge one of them holds.
    """
    frame = pd.read_csv(path, sep="\t")
    for column in ("subject", "object"):
        frame[f"{column}_id"] = frame[column].map(index)
    missing = frame[frame.subject_id.isna() | frame.object_id.isna()]
    if len(missing):
        print(f"  {len(missing):,} split pairs are not in this node table; dropped", flush=True)
    frame = frame.dropna(subset=["subject_id", "object_id"]).astype(
        {
            "subject_id": np.int64,
            "object_id": np.int64,
        }
    )
    every_pair = frame[["subject_id", "object_id"]].to_numpy()
    shared = frame[(frame.in_kg == 1) & (frame.in_pharos == 1)]
    return shared, every_pair, np.unique(every_pair)


def pair_key(subject: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return subject.astype(np.int64) * graph.num_nodes + obj.astype(np.int64)


if args.split:
    config.require(args.split)
    shared, every_pair, _ = load_shared_split(args.split)
    subject_pool = np.unique(shared.subject_id.to_numpy())
    object_pool = np.unique(shared.object_id.to_numpy())
    forbidden = set(pair_key(every_pair[:, 0], every_pair[:, 1]).tolist())
    print(
        f"shared split: {len(shared):,} pairs, pools {len(subject_pool):,} subjects / "
        f"{len(object_pool):,} objects",
        flush=True,
    )
else:
    subject_pool, object_pool = kgraph.side_pools(graph.edges, TARGET)
    target_rows = graph.edges[graph.edges[:, 1] == TARGET]
    forbidden = set(pair_key(target_rows[:, 0], target_rows[:, 2]).tolist())
    print(f"own-graph split on {len(target_rows):,} {args.predicate} edges", flush=True)


def negatives_for(positives: np.ndarray, seed: int) -> np.ndarray:
    """One corrupted pair per positive, half each side, type-matched and filtered."""
    rng = np.random.default_rng(90000 + seed)
    out = np.empty((len(positives), 3), dtype=np.int64)
    for row, (subject, _predicate, obj) in enumerate(positives):
        candidate = (int(subject), int(obj))
        for _attempt in range(100):
            if rng.random() < 0.5:
                candidate = (int(subject_pool[rng.integers(len(subject_pool))]), int(obj))
            else:
                candidate = (int(subject), int(object_pool[rng.integers(len(object_pool))]))
            if candidate[0] != candidate[1] and (
                candidate[0] * graph.num_nodes + candidate[1] not in forbidden
            ):
                break
        out[row] = (candidate[0], TARGET, candidate[1])
    return out


def positives_and_training_edges(seed: int):
    """(train, val, test) target triples, and the message-passing edges for this seed.

    The validation and test pairs are removed from the message-passing edges. Aggregating over
    an edge and then asking the model to predict it is leakage; it shows up as an AUC near 1.
    """
    if args.split:
        column = f"split_seed{seed}"
        if column not in shared.columns:
            raise SystemExit(f"{args.split} has no {column}; re-run pharos_split.py --seeds")
        parts = {}
        for name in ("train", "val", "test"):
            rows = shared[shared[column] == name]
            parts[name] = np.column_stack(
                [
                    rows.subject_id.to_numpy(),
                    np.full(len(rows), TARGET, dtype=np.int64),
                    rows.object_id.to_numpy(),
                ]
            ).astype(np.int64)
    else:
        rows = graph.edges[graph.edges[:, 1] == TARGET].astype(np.int64)
        train_i, val_i, test_i = kgraph.split_rows(
            len(rows), seed, train=args.train_frac, val=args.val_frac
        )
        parts = {"train": rows[train_i], "val": rows[val_i], "test": rows[test_i]}

    held_out = np.concatenate([parts["val"], parts["test"]])
    remove = pair_key(held_out[:, 0], held_out[:, 2])
    is_target = graph.edges[:, 1] == TARGET
    keys = pair_key(graph.edges[:, 0], graph.edges[:, 2])
    drop = is_target & np.isin(keys, remove)
    return parts, graph.edges[~drop]


# ------------------------------------------------------------------ one seed


def score_in_batches(model, x, indptr, indices, triples, rng, batch=8192) -> np.ndarray:
    """Scores for `triples`, each batch on its own sampled neighbourhood."""
    model.eval()
    out = np.empty(len(triples), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(triples), batch):
            chunk = triples[start : start + batch]
            seeds = np.unique(np.concatenate([chunk[:, 0], chunk[:, 2]]))
            nodes, edge_index, _ = sage.sample_subgraph(indptr, indices, seeds, EVAL_FANOUTS, rng)
            h = model.encode(
                torch.as_tensor(x[nodes]).to(DEVICE),
                edge_index.to(DEVICE),
                torch.as_tensor(nodes).long().to(DEVICE),
            )
            local = np.column_stack(
                [
                    np.searchsorted(nodes, chunk[:, 0]),
                    chunk[:, 1],
                    np.searchsorted(nodes, chunk[:, 2]),
                ]
            )
            out[start : start + batch] = (
                model.score(h, torch.as_tensor(local).long().to(DEVICE)).cpu().numpy()
            )
    return out


def run_seed(seed: int):
    parts, train_edges = positives_and_training_edges(seed)
    train_pos, val_pos, test_pos = parts["train"], parts["val"], parts["test"]
    if len(test_pos) < MIN_SUBSET:
        raise SystemExit(f"seed {seed}: only {len(test_pos)} test positives")

    val_neg = negatives_for(val_pos, seed * 7 + 1)
    test_neg = negatives_for(test_pos, seed * 7 + 2)

    x = kgraph.features(graph.category, graph.category_sets, train_edges, graph.num_nodes)
    indptr, indices = kgraph.csr(train_edges, graph.num_nodes)
    rng = np.random.default_rng(5000 + seed)

    torch.manual_seed(1000 + seed)
    model = sage.LinkPredictor(
        in_dim=x.shape[1],
        num_predicates=graph.num_predicates,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        num_nodes=graph.num_nodes,
        embed_dim=args.embed_dim,
    ).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_roc, best_state, since_best = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_pos))
        # fresh negatives every epoch: against a fixed set the model memorises the corruptions
        epoch_neg = negatives_for(train_pos, seed * 100003 + epoch)
        total_loss, steps = 0.0, 0
        for start in range(0, len(order), args.batch):
            rows = order[start : start + args.batch]
            batch_pos, batch_neg = train_pos[rows], epoch_neg[rows]
            triples = np.concatenate([batch_pos, batch_neg])
            seeds = np.unique(np.concatenate([triples[:, 0], triples[:, 2]]))
            nodes, edge_index, _ = sage.sample_subgraph(indptr, indices, seeds, FANOUTS, rng)
            local = np.column_stack(
                [
                    np.searchsorted(nodes, triples[:, 0]),
                    triples[:, 1],
                    np.searchsorted(nodes, triples[:, 2]),
                ]
            )
            optimiser.zero_grad()
            h = model.encode(
                torch.as_tensor(x[nodes]).to(DEVICE),
                edge_index.to(DEVICE),
                torch.as_tensor(nodes).long().to(DEVICE),
            )
            logits = model.score(h, torch.as_tensor(local).long().to(DEVICE))
            labels = torch.cat([torch.ones(len(batch_pos)), torch.zeros(len(batch_neg))]).to(DEVICE)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += float(loss)
            steps += 1

        roc = metrics(
            score_in_batches(model, x, indptr, indices, val_pos, rng),
            score_in_batches(model, x, indptr, indices, val_neg, rng),
        )["roc_auc"]
        print(
            f"  seed {seed} epoch {epoch:3d}  loss {total_loss / max(steps, 1):.4f}  "
            f"val roc {roc:.3f}",
            flush=True,
        )
        if roc > best_roc:
            best_roc, since_best = roc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    scores = {
        "gnn_pos": score_in_batches(model, x, indptr, indices, test_pos, rng),
        "gnn_neg": score_in_batches(model, x, indptr, indices, test_neg, rng),
        "degree_pos": kgraph.degree_scores(test_pos, train_edges, graph.num_nodes),
        "degree_neg": kgraph.degree_scores(test_neg, train_edges, graph.num_nodes),
    }
    return scores, kgraph.both_seen(test_pos, train_edges), best_roc


# ------------------------------------------------------------------ every seed

rows, seen_rows = [], []
started = time.time()
for seed in range(args.seeds):
    scores, mask, best_val = run_seed(seed)
    for name in ("gnn", "degree"):
        rows.append(
            {"seed": seed, "model": name, **metrics(scores[f"{name}_pos"], scores[f"{name}_neg"])}
        )
        for label, subset in (("both_seen", mask), ("has_unseen", ~mask)):
            if subset.sum() >= MIN_SUBSET:
                seen_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "subset": label,
                        "n_pos": int(subset.sum()),
                        "roc_auc": metrics(scores[f"{name}_pos"][subset], scores[f"{name}_neg"])[
                            "roc_auc"
                        ],
                    }
                )
    print(
        f"seed {seed} done, best val roc {best_val:.3f} ({time.time() - started:.0f}s)",
        flush=True,
    )

per_seed = pd.DataFrame(rows)
summary = (
    per_seed.groupby("model")
    .agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        roc_min=("roc_auc", "min"),
        roc_max=("roc_auc", "max"),
        pr_mean=("pr_auc", "mean"),
        pr_std=("pr_auc", "std"),
    )
    .sort_values("roc_mean", ascending=False)
    .reset_index()
)
prevalence = float(per_seed.prevalence.mean())

print(f"\n=== {args.graph}, {args.predicate}, {args.seeds} independent splits ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nROC AUC 0.500 = chance.")
print(
    f"PR AUC baseline (prevalence of the positive class) = {prevalence:.3f} -- any PR AUC near "
    "this is chance, not skill."
)
if len(summary) == 2:
    print(
        f"gap, {summary.model.iloc[0]} over {summary.model.iloc[1]}: "
        f"{summary.roc_mean.iloc[0] - summary.roc_mean.iloc[1]:+.3f}"
    )

seen_unseen = pd.DataFrame(seen_rows)
print("\n=== test edges split by whether both endpoints were in the training edges ===")
if seen_unseen.empty:
    print(f"  no subset reached {MIN_SUBSET} edges")
else:
    print(
        seen_unseen.groupby(["model", "subset"])
        .roc_auc.agg(["mean", "std", "count"])
        .reset_index()
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

if not args.no_write:
    per_seed.to_csv(outputs[0], sep="\t", index=False)
    summary.to_csv(outputs[1], sep="\t", index=False)
    seen_unseen.to_csv(outputs[2], sep="\t", index=False)
    outputs[3].write_text(
        json.dumps(
            {
                "args": vars(args),
                "graph": {
                    "tag": args.graph,
                    "num_nodes": graph.num_nodes,
                    "num_edges": graph.num_edges,
                    "num_predicates": graph.num_predicates,
                    "dropped": graph.dropped,
                },
                "prevalence": prevalence,
                "models": ["gnn", "degree"],
                "torch": torch.__version__,
                "device": str(DEVICE),
                "python": platform.python_version(),
                "seconds": round(time.time() - started, 1),
                "note": "negatives are type-matched and filtered; with --split they are "
                "derived from the split file alone, so both graphs draw identical ones. "
                "Message passing runs over training edges only.",
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {outputs[1]}")
