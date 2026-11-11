"""The merged graph's loading and layout, written once.

`rotate.py` holds the embedding algebra so that detect and evaluate cannot drift on it; this
holds the graph layout for the same reason, so that `kg_prepare.py`, `pharos_split.py` and
`gnn.py` cannot disagree about what a node id means.

Nothing here imports torch, so the splits and the sampling are testable in the repo's own
environment.

The layout:

    node_ids        (N,) CURIEs, position = node id
    category        (N,) int32, an index into `category_sets`
    category_sets   the distinct category lists the graph actually contains -- a few hundred,
                    not N. The same trick KGB-CIA's cleaning agent uses, for the same reason:
                    ORION writes every ancestor, so the lists are long and highly repeated.
    edges           (E, 3) int32 of (subject, predicate, object), deduplicated

Two things the loader does on purpose, because link prediction is wrong without them:

    dedup           the same (subject, predicate, object) from two sources is ONE triple. Left
                    in, the duplicate lands in train and in test and the model scores it from
                    memory. The merged graph has 22 sources, so this is not hypothetical.
    self loops      dropped. Trivially predictable, and KGB-CIA's cleaning agent flags them.
"""

import json
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KnowledgeGraph:
    node_ids: np.ndarray  # (N,) object array of CURIEs
    category: np.ndarray  # (N,) int32 index into category_sets
    category_sets: list[tuple[str, ...]]
    predicates: list[str]
    edges: np.ndarray  # (E, 3) int32
    dropped: dict[str, int]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_predicates(self) -> int:
        return len(self.predicates)

    def predicate_id(self, name: str) -> int:
        try:
            return self.predicates.index(name)
        except ValueError:
            raise SystemExit(
                f"{name} is not in this graph. Present: {', '.join(sorted(self.predicates)[:20])}"
            ) from None

    def index(self) -> dict[str, int]:
        """CURIE -> node id. Built on demand; 1.2M entries is ~150 MB."""
        return {curie: i for i, curie in enumerate(self.node_ids)}

    def describe(self) -> str:
        return (
            f"{self.num_nodes:,} nodes, {self.num_edges:,} triples, "
            f"{self.num_predicates} predicates, dropped {self.dropped}"
        )


def _jsonl(path: Path):
    with open(path) as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: {exc}") from exc


def read_nodes(path: str | Path) -> tuple[dict[str, int], np.ndarray, np.ndarray, list]:
    """nodes.jsonl -> (index, node_ids, category, category_sets)."""
    index: dict[str, int] = {}
    node_ids: list[str] = []
    set_index: dict[tuple[str, ...], int] = {}
    category_sets: list[tuple[str, ...]] = []
    category = array("i")

    for record in _jsonl(Path(path)):
        curie = record.get("id")
        if not curie or curie in index:
            continue
        index[curie] = len(node_ids)
        node_ids.append(curie)
        key = tuple(record.get("category") or ())
        if key not in set_index:
            set_index[key] = len(category_sets)
            category_sets.append(key)
        category.append(set_index[key])

    return (
        index,
        np.array(node_ids, dtype=object),
        np.frombuffer(category, dtype=np.int32).copy(),
        category_sets,
    )


def read_edges(
    path: str | Path, index: dict[str, int], keep: set[str] | None = None
) -> tuple[np.ndarray, list[str], Counter, Counter]:
    """edges.jsonl -> ((E,3) int32, predicates, census, dropped).

    `keep` restricts to a predicate set; the census counts every line regardless, because the
    pass over the file is the expensive part and a census makes a different choice of predicate
    a one-flag change rather than a re-read. Stage 1 (`extract_edges.py`) does the same.
    """
    predicate_index: dict[str, int] = {}
    predicates: list[str] = []
    census: Counter = Counter()
    dropped: Counter = Counter()
    subject_column, predicate_column, object_column = array("i"), array("i"), array("i")

    for record in _jsonl(Path(path)):
        subject, predicate, obj = (
            record.get("subject"),
            record.get("predicate"),
            record.get("object"),
        )
        if not (subject and predicate and obj):
            dropped["malformed"] += 1
            continue
        census[predicate] += 1
        if keep is not None and predicate not in keep:
            continue
        if subject not in index or obj not in index:
            dropped["dangling"] += 1
            continue
        if subject == obj:
            dropped["self_loop"] += 1
            continue
        if predicate not in predicate_index:
            predicate_index[predicate] = len(predicates)
            predicates.append(predicate)
        subject_column.append(index[subject])
        predicate_column.append(predicate_index[predicate])
        object_column.append(index[obj])

    raw = np.column_stack(
        [
            np.frombuffer(subject_column, dtype=np.int32),
            np.frombuffer(predicate_column, dtype=np.int32),
            np.frombuffer(object_column, dtype=np.int32),
        ]
    )
    edges = np.unique(raw, axis=0) if len(raw) else raw.reshape(0, 3)
    dropped["duplicate_triple"] = len(raw) - len(edges)
    return edges, predicates, census, dropped


def load_jsonl(
    nodes_path: str | Path, edges_path: str | Path, keep: set[str] | None = None
) -> KnowledgeGraph:
    index, node_ids, category, category_sets = read_nodes(nodes_path)
    edges, predicates, _census, dropped = read_edges(edges_path, index, keep=keep)
    return KnowledgeGraph(node_ids, category, category_sets, predicates, edges, dict(dropped))


def save(path: str | Path, kg: KnowledgeGraph) -> None:
    """Compact arrays plus a json sidecar for the string tables."""
    path = Path(path)
    np.savez_compressed(path, edges=kg.edges, category=kg.category)
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "node_ids": kg.node_ids.tolist(),
                "category_sets": [list(s) for s in kg.category_sets],
                "predicates": kg.predicates,
                "dropped": kg.dropped,
            }
        )
    )


def load(path: str | Path) -> KnowledgeGraph:
    path = Path(path)
    arrays = np.load(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return KnowledgeGraph(
        np.array(meta["node_ids"], dtype=object),
        arrays["category"],
        [tuple(s) for s in meta["category_sets"]],
        meta["predicates"],
        arrays["edges"],
        meta["dropped"],
    )


# ------------------------------------------------------------------ adjacency, features, splits


def csr(edges: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric neighbour lists as (indptr, indices), built from the TRAINING edges only.

    Message passing over a validation or test edge and then asking the model to predict that
    edge is leakage, and it shows up as an AUC near 1. Every caller passes a split, never the
    whole graph.
    """
    source = np.concatenate([edges[:, 0], edges[:, 2]]).astype(np.int64)
    target = np.concatenate([edges[:, 2], edges[:, 0]]).astype(np.int64)
    order = np.argsort(source, kind="stable")
    indices = target[order]
    counts = np.bincount(source, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices


def degrees(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """Undirected degree over the given edge subset -- pass the TRAINING edges."""
    out = np.zeros(num_nodes, dtype=np.float32)
    if len(edges):
        np.add.at(out, edges[:, 0].astype(np.int64), 1.0)
        np.add.at(out, edges[:, 2].astype(np.int64), 1.0)
    return out


def features(
    category: np.ndarray,
    category_sets: list[tuple[str, ...]],
    train_edges: np.ndarray,
    num_nodes: int,
    max_categories: int = 48,
) -> np.ndarray:
    """(N, F) float32: multi-hot over the commonest categories, plus log1p training degree.

    Deliberately NOT a learned per-node embedding. An embedding table is transductive: a node
    with no training edge keeps its random row, and the unseen-endpoint number -- the one that
    showed RotatE falling from 0.881 to 0.685 -- stops meaning anything. Categories and degree
    exist for every node, seen or not.
    """
    counts: Counter = Counter()
    for set_id, n in Counter(category.tolist()).items():
        for name in category_sets[set_id]:
            counts[name] += n
    kept = [name for name, _ in counts.most_common(max_categories)]
    position = {name: i for i, name in enumerate(kept)}

    # one row per distinct category list, then one gather: the lists repeat heavily
    per_set = np.zeros((len(category_sets), len(kept)), dtype=np.float32)
    for set_id, names in enumerate(category_sets):
        for name in names:
            if name in position:
                per_set[set_id, position[name]] = 1.0

    degree = degrees(train_edges, num_nodes)
    return np.column_stack(
        [per_set[category], np.log1p(degree), (degree == 0).astype(np.float32)]
    ).astype(np.float32)


def split_rows(num_rows: int, seed: int, train: float = 0.8, val: float = 0.1):
    """One independent permutation into train / validation / test row indices."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError(f"train={train} val={val} leaves no test rows")
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(num_rows)
    n_train, n_val = int(train * num_rows), int(val * num_rows)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def sample_negatives(
    positives: np.ndarray,
    known: set[tuple[int, int, int]],
    subject_pool: np.ndarray,
    object_pool: np.ndarray,
    seed: int,
) -> np.ndarray:
    """One corrupted triple per positive, half subject-side, half object-side.

    Two things this gets right that a naive corruption does not:

      filtered    a draw that is already a known triple is redrawn, so no "negative" is an
                  edge the graph asserts.
      type-matched
                  corruptions come from the pool of nodes that actually appear on that side of
                  that predicate, not from all 1.2M nodes. Drawn uniformly, a gene-disease
                  negative is overwhelmingly gene-gene, and every model separates that on
                  category alone -- the AUC then measures a type check, not link prediction.
    """
    rng = np.random.default_rng(90000 + seed)
    out = np.empty((len(positives), 3), dtype=np.int32)
    for row, (subject, predicate, obj) in enumerate(positives):
        candidate = (int(subject), int(predicate), int(obj))
        for _attempt in range(100):
            if rng.random() < 0.5:
                candidate = (
                    int(subject_pool[rng.integers(len(subject_pool))]),
                    int(predicate),
                    int(obj),
                )
            else:
                candidate = (
                    int(subject),
                    int(predicate),
                    int(object_pool[rng.integers(len(object_pool))]),
                )
            if candidate[0] != candidate[2] and candidate not in known:
                break
        out[row] = candidate
    return out


def side_pools(edges: np.ndarray, predicate_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The nodes that appear as subject and as object of this predicate: the corruption pools."""
    rows = edges[edges[:, 1] == predicate_id]
    return np.unique(rows[:, 0]), np.unique(rows[:, 2])


def both_seen(test_positives: np.ndarray, train_edges: np.ndarray) -> np.ndarray:
    """Mask: both endpoints of this test triple appear in the training edges."""
    seen = np.zeros(0, dtype=bool)
    if len(train_edges):
        top = int(max(train_edges[:, 0].max(), train_edges[:, 2].max())) + 1
        seen = np.zeros(top, dtype=bool)
        seen[train_edges[:, 0]] = True
        seen[train_edges[:, 2]] = True

    def known(node: int) -> bool:
        return node < len(seen) and bool(seen[node])

    return np.array([known(int(s)) and known(int(o)) for s, _, o in test_positives], dtype=bool)


def degree_scores(triples: np.ndarray, train_edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """The null hypothesis, as AGENTS.md requires beside every flag: a plausible edge is just
    an edge between two well-connected nodes. Garbage-Collector's degree row, sign flipped --
    there the question was "is this edge wrong", here it is "does this edge exist".
    """
    degree = np.log1p(degrees(train_edges, num_nodes))
    return degree[triples[:, 0].astype(np.int64)] + degree[triples[:, 2].astype(np.int64)]
