"""Stage 5c. GraphSAGE link prediction on a prepared graph, with the baseline and the spread.

Two objectives, and the choice matters more than any hyperparameter here:

    --objective supervised      encoder and decoder trained end to end on the target
                                predicate's edges alone. With the shared split that is 12,367
                                edges, so at --batch 4096 it is FOUR optimiser steps an epoch;
                                the first runs stopped after ~35 steps total, with a 1.2M x 64
                                embedding table still at its random initialisation.
    --objective unsupervised    the encoder is trained first on ALL of this graph's own edges
                                (link reconstruction, every predicate), then FROZEN, and only a
                                small decoder is fitted on the overlapping pairs. The encoder
                                sees 11.4M edges instead of 12k.

Neither is assumed to win. On a synthetic control where the signal is reachable from both,
they tie (0.948 unsupervised, 0.953 supervised at a matched step budget) while a degree-only
baseline stays at 0.528 -- so the control proves the plumbing, not the ranking. Which one wins
on a real graph is a measurement, and the budget has to be matched before the comparison means
anything.

Two ways to split it:

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
ap.add_argument(
    "--objective",
    choices=("supervised", "unsupervised"),
    default="supervised",
    help="supervised: encoder and decoder trained end to end on the target predicate alone. "
    "With a shared split that is ~12k edges and 4 optimiser steps an epoch, so the encoder "
    "barely moves off its initialisation. unsupervised: the encoder is trained first on ALL "
    "of this graph's own edges (link reconstruction), then FROZEN, and only a small decoder "
    "is fitted on the overlapping pairs.",
)
ap.add_argument("--encoder-steps", type=int, default=2000, help="unsupervised: optimiser steps")
ap.add_argument("--encoder-batch", type=int, default=4096, help="unsupervised: edges per step")
ap.add_argument("--encoder-eval-every", type=int, default=200)
ap.add_argument("--encoder-patience", type=int, default=5, help="evaluations without a gain")
ap.add_argument("--decoder", choices=("distmult", "mlp"), default="distmult")
ap.add_argument("--decoder-epochs", type=int, default=200)
ap.add_argument("--decoder-lr", type=float, default=0.01)
ap.add_argument("--decoder-patience", type=int, default=20)
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


def load_shared_split(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shared pairs as node ids, the whole file as written).

    The whole file is returned as well because the negative sampler works in the file's own
    CURIE space, not in this graph's node ids -- see `comparison space` below.
    """
    whole = pd.read_csv(path, sep="\t")
    frame = whole.copy()
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
    shared = frame[(frame.in_kg == 1) & (frame.in_pharos == 1)]
    return shared, whole


def pair_key(subject: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return subject.astype(np.int64) * graph.num_nodes + obj.astype(np.int64)


# ---------------------------------------------------------------- comparison space
#
# Negatives are drawn in a space that does NOT depend on the graph being scored, so that two
# graphs prepared against different node tables draw the IDENTICAL negative pairs from a seed.
#
# With --split that space is the split file's own CURIE index: its pools are the shared pairs'
# distinct subjects and objects ordered by CURIE, and its filter is every pair the file lists.
# Both are properties of the file alone, so both runs see them identically.
#
# What this replaces: pools built with np.unique over NODE IDS, which sort differently in each
# graph. The same CURIE set came out in a different order, so the same RNG stream picked
# different entities -- measured on the 2026-11-12 runs, only 0-2 of each seed's 1,558 test
# negatives coincided between the two graphs. The filter was also built through each graph's
# own node table, so a pair the graph could not map was silently absent from it, and a
# rejection in one graph but not the other desynchronised the RNG stream for every later row.
#
# Without --split only one graph is ever involved, so comparison space is its own node ids.

if args.split:
    config.require(args.split)
    shared, whole = load_shared_split(args.split)
    curies = sorted(set(whole.subject) | set(whole.object))
    comparison = {curie: i for i, curie in enumerate(curies)}
    SPAN = len(curies)
    FORBIDDEN = {
        comparison[s] * SPAN + comparison[o] for s, o in zip(whole.subject, whole.object)
    }
    POOL_S = np.array(sorted(comparison[c] for c in shared.subject.unique()), dtype=np.int64)
    POOL_O = np.array(sorted(comparison[c] for c in shared.object.unique()), dtype=np.int64)
    TO_NODE = np.array([index.get(c, -1) for c in curies], dtype=np.int64)
    TO_COMPARISON = np.full(graph.num_nodes, -1, dtype=np.int64)
    for curie, position in comparison.items():
        node = index.get(curie)
        if node is not None:
            TO_COMPARISON[node] = position
    if (TO_NODE[POOL_S] < 0).any() or (TO_NODE[POOL_O] < 0).any():
        raise SystemExit(
            "a shared pair's endpoint is missing from this graph's node table, which cannot "
            "happen for a pair both graphs assert -- the split and the graph disagree"
        )
    print(
        f"shared split: {len(shared):,} pairs, pools {len(POOL_S):,} subjects / "
        f"{len(POOL_O):,} objects, filter {len(FORBIDDEN):,} pairs "
        f"(drawn in the split's CURIE space, identical for every graph)",
        flush=True,
    )
else:
    POOL_S, POOL_O = kgraph.side_pools(graph.edges, TARGET)
    target_rows = graph.edges[graph.edges[:, 1] == TARGET]
    SPAN = graph.num_nodes
    FORBIDDEN = set(pair_key(target_rows[:, 0], target_rows[:, 2]).tolist())
    TO_NODE = np.arange(graph.num_nodes, dtype=np.int64)
    TO_COMPARISON = np.arange(graph.num_nodes, dtype=np.int64)
    print(f"own-graph split on {len(target_rows):,} {args.predicate} edges", flush=True)


def negatives_for(positives: np.ndarray, seed: int) -> np.ndarray:
    """One corrupted pair per positive, half each side, type-matched and filtered.

    Every draw, every rejection and every retry happens in comparison space, so the sequence
    of candidates depends only on the split file and the seed. Two graphs therefore produce
    byte-identical negatives. Node ids are assigned only at the end.
    """
    rng = np.random.default_rng(90000 + seed)
    out = np.empty((len(positives), 3), dtype=np.int64)
    for row, (subject, _predicate, obj) in enumerate(positives):
        s, o = int(TO_COMPARISON[subject]), int(TO_COMPARISON[obj])
        candidate = (s, o)
        for _attempt in range(100):
            if rng.random() < 0.5:
                candidate = (int(POOL_S[rng.integers(len(POOL_S))]), o)
            else:
                candidate = (s, int(POOL_O[rng.integers(len(POOL_O))]))
            if candidate[0] != candidate[1] and (
                candidate[0] * SPAN + candidate[1] not in FORBIDDEN
            ):
                break
        out[row] = (TO_NODE[candidate[0]], TARGET, TO_NODE[candidate[1]])
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


# ------------------------------------------------------------- unsupervised encoder, frozen


def encoder_training_edges(parts: dict) -> np.ndarray:
    """Every edge of this graph EXCEPT the held-out pairs, under any predicate, either way.

    Stricter than the supervised path's removal, which drops a held-out pair only when it
    carries the target predicate. Here the encoder trains on the edges themselves, so leaving
    a test pair in under some other predicate -- or reversed -- trains the encoder on the very
    pair the decoder is later scored on. That leak reads as a high AUC, not as an error.
    """
    held_out = np.concatenate([parts["val"], parts["test"]])
    remove = np.concatenate(
        [pair_key(held_out[:, 0], held_out[:, 2]), pair_key(held_out[:, 2], held_out[:, 0])]
    )
    keys = pair_key(graph.edges[:, 0], graph.edges[:, 2])
    return graph.edges[~np.isin(keys, remove)]


def corrupt(triples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One negative per edge: replace the subject or the object with any node."""
    out = triples.copy()
    flip = rng.random(len(out)) < 0.5
    replacement = rng.integers(0, graph.num_nodes, size=len(out))
    out[flip, 0] = replacement[flip]
    out[~flip, 2] = replacement[~flip]
    return out


def _forward(model, x, indptr, indices, triples, rng, fanouts):
    seeds = np.unique(np.concatenate([triples[:, 0], triples[:, 2]]))
    nodes, edge_index, _ = sage.sample_subgraph(indptr, indices, seeds, fanouts, rng)
    h = model.encode(
        torch.as_tensor(x[nodes]).to(DEVICE),
        edge_index.to(DEVICE),
        torch.as_tensor(nodes).long().to(DEVICE),
    )
    local = np.column_stack(
        [
            np.searchsorted(nodes, triples[:, 0]),
            triples[:, 1],
            np.searchsorted(nodes, triples[:, 2]),
        ]
    )
    return model.score(h, torch.as_tensor(local).long().to(DEVICE))


def train_encoder(encoder_edges: np.ndarray, seed: int):
    """Link reconstruction over the whole graph. Returns (model, x, indptr, indices, best)."""
    rng = np.random.default_rng(7000 + seed)
    order = rng.permutation(len(encoder_edges))
    n_val = min(20000, max(1, len(encoder_edges) // 100))
    val_rows = encoder_edges[order[:n_val]].astype(np.int64)
    train_rows = encoder_edges[order[n_val:]].astype(np.int64)

    # message passing uses the encoder's edges; its own validation slice is scored, not removed,
    # because at 1% it cannot carry the representation and removing it would cost a CSR rebuild
    x = kgraph.features(graph.category, graph.category_sets, encoder_edges, graph.num_nodes)
    indptr, indices = kgraph.csr(encoder_edges, graph.num_nodes)

    torch.manual_seed(2000 + seed)
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
    val_neg = corrupt(val_rows, np.random.default_rng(7777 + seed))

    best, best_state, since_best, running = -1.0, None, 0, []
    for step in range(1, args.encoder_steps + 1):
        model.train()
        rows = train_rows[rng.integers(0, len(train_rows), size=args.encoder_batch)]
        triples = np.concatenate([rows, corrupt(rows, rng)])
        labels = torch.cat(
            [torch.ones(len(rows)), torch.zeros(len(rows))]
        ).to(DEVICE)
        optimiser.zero_grad()
        loss = loss_fn(_forward(model, x, indptr, indices, triples, rng, FANOUTS), labels)
        loss.backward()
        optimiser.step()
        running.append(float(loss.detach()))

        if step % args.encoder_eval_every == 0:
            model.eval()
            with torch.no_grad():
                pos = _forward(model, x, indptr, indices, val_rows, rng, EVAL_FANOUTS)
                neg = _forward(model, x, indptr, indices, val_neg, rng, EVAL_FANOUTS)
            roc = roc_auc_score(
                np.r_[np.ones(len(val_rows)), np.zeros(len(val_neg))],
                np.r_[pos.cpu().numpy(), neg.cpu().numpy()],
            )
            print(
                f"    encoder step {step:5d}  loss {np.mean(running):.4f}  "
                f"reconstruction roc {roc:.3f}",
                flush=True,
            )
            running = []
            if roc > best:
                best, since_best = roc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= args.encoder_patience:
                    print("    encoder early stop", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x, indptr, indices, best


def embed(model, x, indptr, indices, wanted: np.ndarray, rng, batch=4096) -> torch.Tensor:
    """Frozen per-node representations for `wanted`, as a (len(wanted), hidden) tensor."""
    model.eval()
    out = torch.empty((len(wanted), args.hidden), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for start in range(0, len(wanted), batch):
            chunk = wanted[start : start + batch]
            nodes, edge_index, _ = sage.sample_subgraph(indptr, indices, chunk, EVAL_FANOUTS, rng)
            h = model.encode(
                torch.as_tensor(x[nodes]).to(DEVICE),
                edge_index.to(DEVICE),
                torch.as_tensor(nodes).long().to(DEVICE),
            )
            out[start : start + len(chunk)] = h[np.searchsorted(nodes, chunk)]
    return out


class Decoder(torch.nn.Module):
    """What is fitted on the overlapping pairs, over frozen node representations.

    distmult   one learned relation vector; the same scoring family the encoder trained with,
               so the unsupervised and supervised rows stay comparable.
    mlp        a hidden layer over [h_s, h_o, h_s * h_o], for when the relation is not a
               diagonal bilinear form.
    """

    def __init__(self, dim: int, kind: str, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.kind = kind
        if kind == "distmult":
            self.relation = torch.nn.Parameter(torch.randn(dim) * 0.1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3 * dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )

    def forward(self, hs: torch.Tensor, ho: torch.Tensor) -> torch.Tensor:
        if self.kind == "distmult":
            return (hs * self.relation * ho).sum(-1)
        return self.net(torch.cat([hs, ho, hs * ho], dim=1)).squeeze(-1)


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


def run_seed_supervised(seed: int):
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


def run_seed_unsupervised(seed: int):
    """Encoder on this graph's own edges, frozen; decoder on the overlapping pairs."""
    parts, _ = positives_and_training_edges(seed)
    train_pos, val_pos, test_pos = parts["train"], parts["val"], parts["test"]
    if len(test_pos) < MIN_SUBSET:
        raise SystemExit(f"seed {seed}: only {len(test_pos)} test positives")

    encoder_edges = encoder_training_edges(parts)
    print(
        f"  encoder trains on {len(encoder_edges):,} of {graph.num_edges:,} edges "
        f"({graph.num_edges - len(encoder_edges):,} held out as the decoder's val/test pairs)",
        flush=True,
    )
    model, x, indptr, indices, encoder_roc = train_encoder(encoder_edges, seed)

    train_neg = negatives_for(train_pos, seed * 7 + 3)
    val_neg = negatives_for(val_pos, seed * 7 + 1)
    test_neg = negatives_for(test_pos, seed * 7 + 2)

    # every node the decoder will ever ask about, embedded once from the frozen encoder
    wanted = np.unique(
        np.concatenate(
            [subject_pool, object_pool]
            + [t[:, c] for t in (train_pos, val_pos, test_pos) for c in (0, 2)]
            + [t[:, c] for t in (train_neg, val_neg, test_neg) for c in (0, 2)]
        )
    ).astype(np.int64)
    rng = np.random.default_rng(5000 + seed)
    H = embed(model, x, indptr, indices, wanted, rng)
    print(f"  froze {len(wanted):,} node representations ({args.hidden}d)", flush=True)

    def rows_of(triples: np.ndarray):
        return (
            torch.as_tensor(np.searchsorted(wanted, triples[:, 0])).long().to(DEVICE),
            torch.as_tensor(np.searchsorted(wanted, triples[:, 2])).long().to(DEVICE),
        )

    sets = {
        name: rows_of(t)
        for name, t in (
            ("train_pos", train_pos),
            ("train_neg", train_neg),
            ("val_pos", val_pos),
            ("val_neg", val_neg),
            ("test_pos", test_pos),
            ("test_neg", test_neg),
        )
    }

    decoder = Decoder(args.hidden, args.decoder).to(DEVICE)
    optimiser = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    labels = torch.cat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))]).to(DEVICE)
    val_labels = np.r_[np.ones(len(val_pos)), np.zeros(len(val_neg))]

    def logits(name: str) -> torch.Tensor:
        s, o = sets[name]
        return decoder(H[s], H[o])

    best, best_state, since_best = -1.0, None, 0
    for epoch in range(args.decoder_epochs):
        decoder.train()
        optimiser.zero_grad()
        loss = loss_fn(torch.cat([logits("train_pos"), logits("train_neg")]), labels)
        loss.backward()
        optimiser.step()

        decoder.eval()
        with torch.no_grad():
            scored = torch.cat([logits("val_pos"), logits("val_neg")]).cpu().numpy()
        roc = roc_auc_score(val_labels, scored)
        if epoch % 25 == 0:
            print(
                f"    decoder epoch {epoch:4d}  loss {float(loss.detach()):.4f}  "
                f"val roc {roc:.3f}",
                flush=True,
            )
        if roc > best:
            best, since_best = roc, 0
            best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
        else:
            since_best += 1
            if since_best >= args.decoder_patience:
                break
    if best_state is not None:
        decoder.load_state_dict(best_state)

    decoder.eval()
    with torch.no_grad():
        gnn_pos = logits("test_pos").cpu().numpy()
        gnn_neg = logits("test_neg").cpu().numpy()
    print(f"  encoder reconstruction roc {encoder_roc:.3f}, decoder val roc {best:.3f}", flush=True)
    scores = {
        "gnn_pos": gnn_pos,
        "gnn_neg": gnn_neg,
        "degree_pos": kgraph.degree_scores(test_pos, encoder_edges, graph.num_nodes),
        "degree_neg": kgraph.degree_scores(test_neg, encoder_edges, graph.num_nodes),
    }
    return scores, kgraph.both_seen(test_pos, encoder_edges), best


def run_seed(seed: int):
    if args.objective == "unsupervised":
        return run_seed_unsupervised(seed)
    return run_seed_supervised(seed)


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
