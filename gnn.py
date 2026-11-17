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
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
import kgraph
import linkpred
import sage
from linkpred import MIN_SUBSET

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
ap.add_argument(
    "--encoder-negatives",
    choices=("uniform", "typed"),
    default="uniform",
    help="unsupervised: how the encoder's negatives are drawn, for BOTH its loss and its "
    "validation metric. uniform replaces an endpoint with any node in the graph, which on a "
    "graph this size is almost never a plausible edge -- reconstruction ROC then saturates "
    "above 0.99 within a fifth of the step budget. typed replaces it with a node observed in "
    "that position for that same predicate and rejects draws that are real edges, which is "
    "the kind of negative the decoder is scored on. Reconstruction ROC under typed is NOT "
    "comparable to uniform: it is measured against a harder distribution.",
)
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
outputs = linkpred.output_paths(args.tag)
if not args.no_write:
    config.refuse_overwrite(*outputs)

graph = kgraph.load(npz)
print(graph.describe(), flush=True)

C = linkpred.Comparison(
    graph,
    predicate=args.predicate,
    split=args.split,
    train_frac=args.train_frac,
    val_frac=args.val_frac,
)
TARGET = C.target
metrics = linkpred.metrics


def corrupt(triples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One negative per edge: replace the subject or the object with any node."""
    out = triples.copy()
    flip = rng.random(len(out)) < 0.5
    replacement = rng.integers(0, graph.num_nodes, size=len(out))
    out[flip, 0] = replacement[flip]
    out[~flip, 2] = replacement[~flip]
    return out


EMPTY_KEYS = np.zeros(0, dtype=np.int64)


def _is_true_edge(keys: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Membership against a sorted key array. np.isin would rebuild a hash on every call."""
    if not len(truth):
        return np.zeros(len(keys), dtype=bool)
    position = np.clip(np.searchsorted(truth, keys), 0, len(truth) - 1)
    return truth[position] == keys


class TypedCorruptor:
    """One negative per edge, drawn the way the decoder's are: the replacement comes from the
    nodes observed in that position for that same predicate, and a draw that lands on a real
    edge is redrawn.

    The two edge sets are deliberately different. Pools come from the encoder's OWN edges, so
    nothing about a held-out pair leaks into them. The truth filter comes from ALL of the
    graph's edges, which is the whole point of this class: a held-out pair is by construction
    absent from the encoder's edge set, so without filtering against the full graph it is a
    legal type-matched draw, and the encoder would be trained to score the decoder's own test
    positives as negatives. Uniform corruption never had this problem only because a 1.2M-node
    draw space made it vanishingly unlikely.
    """

    def __init__(self, pool_edges: np.ndarray, filter_edges: np.ndarray):
        self.pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.truth: dict[int, np.ndarray] = {}
        self.unresolved = 0  # negatives still on a real edge after both redraw budgets
        for predicate in np.unique(pool_edges[:, 1]):
            rows = pool_edges[pool_edges[:, 1] == predicate]
            self.pools[int(predicate)] = (np.unique(rows[:, 0]), np.unique(rows[:, 2]))
        for predicate in np.unique(filter_edges[:, 1]):
            rows = filter_edges[filter_edges[:, 1] == predicate]
            self.truth[int(predicate)] = np.unique(C.pair_key(rows[:, 0], rows[:, 2]))

    def __call__(
        self, triples: np.ndarray, rng: np.random.Generator, tries: int = 16
    ) -> np.ndarray:
        # tries is 16, not a handful, because rejection is far harder than the overall density
        # suggests for a hub-heavy predicate. Corrupting the subject against a popular object
        # redraws a real edge with probability indegree(o)/|pool|, which for an ontology root
        # under biolink:subclass_of is ~0.4; measured on pharos319, four tries left 3.6% of
        # that predicate's negatives sitting on real edges, i.e. label noise in the loss.
        # Anything still colliding after 16 falls back to a uniform node, which collides at the
        # graph's overall density and is the v1 behaviour for that row.
        out = triples.copy()
        subject_side = rng.random(len(out)) < 0.5
        for predicate in np.unique(out[:, 1]):
            predicate = int(predicate)
            pools = self.pools.get(predicate)
            if pools is None:
                continue
            truth = self.truth.get(predicate, EMPTY_KEYS)
            of_predicate = out[:, 1] == predicate
            for pool, column, corrupts_subject in ((pools[0], 0, True), (pools[1], 2, False)):
                if not len(pool):
                    continue
                live = np.flatnonzero(of_predicate & (subject_side == corrupts_subject))
                for _ in range(tries):
                    if not len(live):
                        break
                    out[live, column] = pool[rng.integers(0, len(pool), size=len(live))]
                    live = live[_is_true_edge(C.pair_key(out[live, 0], out[live, 2]), truth)]
                for _ in range(tries):
                    if not len(live):
                        break
                    out[live, column] = rng.integers(0, graph.num_nodes, size=len(live))
                    live = live[_is_true_edge(C.pair_key(out[live, 0], out[live, 2]), truth)]
                self.unresolved += len(live)
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
    corrupt_fn = (
        TypedCorruptor(encoder_edges, graph.edges) if args.encoder_negatives == "typed" else corrupt
    )
    val_neg = corrupt_fn(val_rows, np.random.default_rng(7777 + seed))

    best, best_state, since_best, running = -1.0, None, 0, []
    for step in range(1, args.encoder_steps + 1):
        model.train()
        rows = train_rows[rng.integers(0, len(train_rows), size=args.encoder_batch)]
        triples = np.concatenate([rows, corrupt_fn(rows, rng)])
        labels = torch.cat([torch.ones(len(rows)), torch.zeros(len(rows))]).to(DEVICE)
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
    parts, train_edges = C.positives_and_training_edges(seed)
    train_pos, val_pos, test_pos = parts["train"], parts["val"], parts["test"]
    if len(test_pos) < MIN_SUBSET:
        raise SystemExit(f"seed {seed}: only {len(test_pos)} test positives")

    val_neg = C.negatives_for(val_pos, seed * 7 + 1)
    test_neg = C.negatives_for(test_pos, seed * 7 + 2)

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
        epoch_neg = C.negatives_for(train_pos, seed * 100003 + epoch)
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
    parts, _ = C.positives_and_training_edges(seed)
    train_pos, val_pos, test_pos = parts["train"], parts["val"], parts["test"]
    if len(test_pos) < MIN_SUBSET:
        raise SystemExit(f"seed {seed}: only {len(test_pos)} test positives")

    encoder_edges = C.encoder_training_edges(parts)
    print(
        f"  encoder trains on {len(encoder_edges):,} of {graph.num_edges:,} edges "
        f"({graph.num_edges - len(encoder_edges):,} held out as the decoder's val/test pairs), "
        f"{args.encoder_negatives} negatives",
        flush=True,
    )
    model, x, indptr, indices, encoder_roc = train_encoder(encoder_edges, seed)

    train_neg = C.negatives_for(train_pos, seed * 7 + 3)
    val_neg = C.negatives_for(val_pos, seed * 7 + 1)
    test_neg = C.negatives_for(test_pos, seed * 7 + 2)

    # every node the decoder will ever ask about, embedded once from the frozen encoder
    wanted = np.unique(
        np.concatenate(
            [C.to_node[C.pool_s], C.to_node[C.pool_o]]
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
                f"    decoder epoch {epoch:4d}  loss {float(loss.detach()):.4f}  val roc {roc:.3f}",
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
        row, seen = linkpred.score_rows(seed, name, scores, mask)
        rows.append(row)
        seen_rows.extend(seen)
    print(
        f"seed {seed} done, best val roc {best_val:.3f} ({time.time() - started:.0f}s)",
        flush=True,
    )

per_seed, summary, prevalence = linkpred.summarise(rows)
seen_unseen = pd.DataFrame(seen_rows)
linkpred.report(
    f"{args.graph}, {args.predicate}, {args.seeds} independent splits",
    summary,
    prevalence,
    seen_unseen,
)

if not args.no_write:
    linkpred.write(
        outputs,
        per_seed,
        summary,
        seen_unseen,
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
    )
