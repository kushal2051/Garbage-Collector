"""The comparison protocol stage 5 is measured under, written once.

What a link-prediction number means here is fixed by four things that have nothing to do with
the model: which pairs are held out, which negatives they are scored against, which edges the
model was allowed to see, and what the degree baseline on those edges is. Two models are
comparable exactly when all four agree row for row -- so they live here, and `gnn.py` (GraphSAGE
and a decoder) and `rotate_kge.py` (RotatE) both import them rather than each keeping a copy.

That is not a style preference. The negative sampler was WRONG until 2026-11-12 in a way that
produced plausible-looking numbers: pools built over node ids sort differently in each graph, so
the same RNG stream drew different entities and only 0-2 of each seed's 1,558 negatives
coincided between the two runs being compared. A second copy of that code is a second chance to
make the same class of mistake silently.

Nothing about a model belongs in this file.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import config
import kgraph

MIN_SUBSET = 20  # a seen/unseen subset smaller than this is not reported


def output_paths(tag: str):
    """The four files every stage-5 run writes, under one prefix."""
    return [
        config.OUT / f"{tag}{suffix}"
        for suffix in ("_per_seed.tsv", "_summary.tsv", "_seen_unseen.tsv", "_meta.json")
    ]


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


class Comparison:
    """The split, the negatives and the training edges for one graph under one protocol.

    With `split` this is the shared split from pharos_split.py and the comparison is across
    graphs; without it the graph is split on its own target edges and only one graph is ever
    involved.
    """

    def __init__(
        self,
        graph,
        predicate: str,
        split: str | None = None,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        verbose: bool = True,
    ) -> None:
        self.graph = graph
        self.predicate = predicate
        self.split = split
        self.train_frac, self.val_frac = train_frac, val_frac
        self.target = graph.predicate_id(predicate)
        self.index = graph.index()

        # ---------------------------------------------------------- comparison space
        #
        # Negatives are drawn in a space that does NOT depend on the graph being scored, so that
        # two graphs prepared against different node tables draw the IDENTICAL negative pairs
        # from a seed.
        #
        # With a split that space is the split file's own CURIE index: its pools are the shared
        # pairs' distinct subjects and objects ordered by CURIE, and its filter is every pair the
        # file lists. Both are properties of the file alone, so both runs see them identically.
        #
        # What this replaces: pools built with np.unique over NODE IDS, which sort differently in
        # each graph. The same CURIE set came out in a different order, so the same RNG stream
        # picked different entities -- measured on the 2026-11-12 runs, only 0-2 of each seed's
        # 1,558 test negatives coincided between the two graphs. The filter was also built
        # through each graph's own node table, so a pair the graph could not map was silently
        # absent from it, and a rejection in one graph but not the other desynchronised the RNG
        # stream for every later row.
        #
        # Without a split only one graph is ever involved, so comparison space is its own ids.
        if split:
            config.require(split)
            self.shared, self.whole = self._load_shared_split(split, verbose)
            curies = sorted(set(self.whole.subject) | set(self.whole.object))
            comparison = {curie: i for i, curie in enumerate(curies)}
            self.span = len(curies)
            self.forbidden = {
                comparison[s] * self.span + comparison[o]
                for s, o in zip(self.whole.subject, self.whole.object, strict=False)
            }
            self.pool_s = np.array(
                sorted(comparison[c] for c in self.shared.subject.unique()), dtype=np.int64
            )
            self.pool_o = np.array(
                sorted(comparison[c] for c in self.shared.object.unique()), dtype=np.int64
            )
            self.to_node = np.array([self.index.get(c, -1) for c in curies], dtype=np.int64)
            self.to_comparison = np.full(graph.num_nodes, -1, dtype=np.int64)
            for curie, position in comparison.items():
                node = self.index.get(curie)
                if node is not None:
                    self.to_comparison[node] = position
            if (self.to_node[self.pool_s] < 0).any() or (self.to_node[self.pool_o] < 0).any():
                raise SystemExit(
                    "a shared pair's endpoint is missing from this graph's node table, which "
                    "cannot happen for a pair both graphs assert -- the split and the graph "
                    "disagree"
                )
            if verbose:
                print(
                    f"shared split: {len(self.shared):,} pairs, pools {len(self.pool_s):,} "
                    f"subjects / {len(self.pool_o):,} objects, filter {len(self.forbidden):,} "
                    "pairs (drawn in the split's CURIE space, identical for every graph)",
                    flush=True,
                )
        else:
            self.shared = self.whole = None
            self.pool_s, self.pool_o = kgraph.side_pools(graph.edges, self.target)
            target_rows = graph.edges[graph.edges[:, 1] == self.target]
            self.span = graph.num_nodes
            self.forbidden = set(self.pair_key(target_rows[:, 0], target_rows[:, 2]).tolist())
            self.to_node = np.arange(graph.num_nodes, dtype=np.int64)
            self.to_comparison = np.arange(graph.num_nodes, dtype=np.int64)
            if verbose:
                print(f"own-graph split on {len(target_rows):,} {predicate} edges", flush=True)

    # ------------------------------------------------------------------ the split

    def _load_shared_split(self, path: str, verbose: bool):
        """(shared pairs as node ids, the whole file as written).

        The whole file is returned as well because the negative sampler works in the file's own
        CURIE space, not in this graph's node ids -- see `comparison space` above.
        """
        whole = pd.read_csv(path, sep="\t")
        frame = whole.copy()
        for column in ("subject", "object"):
            frame[f"{column}_id"] = frame[column].map(self.index)
        missing = frame[frame.subject_id.isna() | frame.object_id.isna()]
        if len(missing) and verbose:
            print(f"  {len(missing):,} split pairs are not in this node table; dropped", flush=True)
        frame = frame.dropna(subset=["subject_id", "object_id"]).astype(
            {
                "subject_id": np.int64,
                "object_id": np.int64,
            }
        )
        shared = frame[(frame.in_kg == 1) & (frame.in_pharos == 1)]
        return shared, whole

    def pair_key(self, subject: np.ndarray, obj: np.ndarray) -> np.ndarray:
        return subject.astype(np.int64) * self.graph.num_nodes + obj.astype(np.int64)

    # ------------------------------------------------------------------ the negatives

    def negatives_for(self, positives: np.ndarray, seed: int) -> np.ndarray:
        """One corrupted pair per positive, half each side, type-matched and filtered.

        Every draw, every rejection and every retry happens in comparison space, so the sequence
        of candidates depends only on the split file and the seed. Two graphs therefore produce
        byte-identical negatives. Node ids are assigned only at the end.
        """
        rng = np.random.default_rng(90000 + seed)
        out = np.empty((len(positives), 3), dtype=np.int64)
        for row, (subject, _predicate, obj) in enumerate(positives):
            s, o = int(self.to_comparison[subject]), int(self.to_comparison[obj])
            candidate = (s, o)
            for _attempt in range(100):
                if rng.random() < 0.5:
                    candidate = (int(self.pool_s[rng.integers(len(self.pool_s))]), o)
                else:
                    candidate = (s, int(self.pool_o[rng.integers(len(self.pool_o))]))
                if candidate[0] != candidate[1] and (
                    candidate[0] * self.span + candidate[1] not in self.forbidden
                ):
                    break
            out[row] = (self.to_node[candidate[0]], self.target, self.to_node[candidate[1]])
        return out

    # ------------------------------------------------------------------ the edges

    def positives_and_training_edges(self, seed: int):
        """(train, val, test) target triples, and the message-passing edges for this seed.

        The validation and test pairs are removed from the message-passing edges. Aggregating
        over an edge and then asking the model to predict it is leakage; it shows up as an AUC
        near 1.
        """
        graph = self.graph
        if self.split:
            column = f"split_seed{seed}"
            if column not in self.shared.columns:
                raise SystemExit(f"{self.split} has no {column}; re-run pharos_split.py --seeds")
            parts = {}
            for name in ("train", "val", "test"):
                rows = self.shared[self.shared[column] == name]
                parts[name] = np.column_stack(
                    [
                        rows.subject_id.to_numpy(),
                        np.full(len(rows), self.target, dtype=np.int64),
                        rows.object_id.to_numpy(),
                    ]
                ).astype(np.int64)
        else:
            rows = graph.edges[graph.edges[:, 1] == self.target].astype(np.int64)
            train_i, val_i, test_i = kgraph.split_rows(
                len(rows), seed, train=self.train_frac, val=self.val_frac
            )
            parts = {"train": rows[train_i], "val": rows[val_i], "test": rows[test_i]}

        held_out = np.concatenate([parts["val"], parts["test"]])
        remove = self.pair_key(held_out[:, 0], held_out[:, 2])
        is_target = graph.edges[:, 1] == self.target
        keys = self.pair_key(graph.edges[:, 0], graph.edges[:, 2])
        drop = is_target & np.isin(keys, remove)
        return parts, graph.edges[~drop]

    def encoder_training_edges(self, parts: dict) -> np.ndarray:
        """Every edge of this graph EXCEPT the held-out pairs, under any predicate, either way.

        Stricter than the supervised path's removal, which drops a held-out pair only when it
        carries the target predicate. Here the model trains on the edges themselves, so leaving
        a test pair in under some other predicate -- or reversed -- trains it on the very pair it
        is later scored on. That leak reads as a high AUC, not as an error.
        """
        graph = self.graph
        held_out = np.concatenate([parts["val"], parts["test"]])
        remove = np.concatenate(
            [
                self.pair_key(held_out[:, 0], held_out[:, 2]),
                self.pair_key(held_out[:, 2], held_out[:, 0]),
            ]
        )
        keys = self.pair_key(graph.edges[:, 0], graph.edges[:, 2])
        return graph.edges[~np.isin(keys, remove)]


# ------------------------------------------------------------------ results


def score_rows(seed: int, model: str, scores: dict, mask: np.ndarray):
    """One overall metric row for this model and seed, plus its seen/unseen rows.

    `scores` holds `<model>_pos` and `<model>_neg`; `mask` is True where both endpoints of a
    test positive appear in the training edges. The negatives are not subsetted: a subset's ROC
    asks how the positives it contains rank against the whole negative pool.
    """
    positive, negative = scores[f"{model}_pos"], scores[f"{model}_neg"]
    row = {"seed": seed, "model": model, **metrics(positive, negative)}
    seen_rows = []
    for label, subset in (("both_seen", mask), ("has_unseen", ~mask)):
        if subset.sum() >= MIN_SUBSET:
            seen_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "subset": label,
                    "n_pos": int(subset.sum()),
                    "roc_auc": metrics(positive[subset], negative)["roc_auc"],
                }
            )
    return row, seen_rows


def summarise(rows: list[dict]):
    """(per_seed, summary, prevalence) -- mean, spread and range of ROC across the seeds."""
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
    return per_seed, summary, float(per_seed.prevalence.mean())


def report(header: str, summary: pd.DataFrame, prevalence: float, seen_unseen: pd.DataFrame):
    """Every number beside the two things without which it cannot be read, as AGENTS.md asks."""
    print(f"\n=== {header} ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nROC AUC 0.500 = chance.")
    print(
        f"PR AUC baseline (prevalence of the positive class) = {prevalence:.3f} -- any PR AUC "
        "near this is chance, not skill."
    )
    if len(summary) == 2:
        print(
            f"gap, {summary.model.iloc[0]} over {summary.model.iloc[1]}: "
            f"{summary.roc_mean.iloc[0] - summary.roc_mean.iloc[1]:+.3f}"
        )

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


def write(outputs, per_seed, summary, seen_unseen, meta: dict):
    per_seed.to_csv(outputs[0], sep="\t", index=False)
    summary.to_csv(outputs[1], sep="\t", index=False)
    seen_unseen.to_csv(outputs[2], sep="\t", index=False)
    outputs[3].write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {outputs[1]}")
