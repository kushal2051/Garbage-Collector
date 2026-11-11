"""Stage 5b. One train/validation/test split, shared by the merged graph and by Pharos.

The comparison KGB-CIA's evaluation pipeline exists to make is "our graph against the Pharos
baseline", and its one hard requirement is that both runs use an identical train/test edge
split -- otherwise the two numbers are not comparable and the comparison means nothing.

"Identical" is underdefined across two graphs that do not hold the same edges, so this is the
construction used here, stated so it can be argued with:

    shared positives   node pairs carrying the target predicate in BOTH graphs. Only these can
                       be split identically, because only these exist in both.
    the split          computed once, over those shared pairs, from one seed. Both models get
                       the same train, validation and test rows.
    each model's input each graph keeps all of its own other edges for message passing, minus
                       the shared validation and test pairs, which both must hold out.
    negatives          drawn once, type-matched, filtered against the union of both graphs'
                       edges, so a "negative" is not an edge that one of the two asserts.

Pairs that only one graph has are written out too, with `in_kg` / `in_pharos` flags, because
how many there are is the first thing to look at: if the shared set is tiny, this predicate
cannot carry the comparison and the result to report is that fact.

Usage:
    python pharos_split.py --predicate biolink:genetically_associated_with
"""

import argparse
import json

import numpy as np

import config
import kgraph

ap = argparse.ArgumentParser()
ap.add_argument("--predicate", default="biolink:genetically_associated_with")
ap.add_argument("--seeds", type=int, default=10, help="how many independent splits to write")
ap.add_argument("--train-frac", type=float, default=0.8)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--tag", default=None, help="output name (default: the predicate's local name)")
args = ap.parse_args()

tag = args.tag or args.predicate.split(":")[-1]
kg_npz, pharos_npz = config.OUT / "kg.npz", config.OUT / "pharos.npz"
config.require(kg_npz, pharos_npz)
pairs_tsv = config.OUT / f"split_{tag}_pairs.tsv"
meta_json = config.OUT / f"split_{tag}_meta.json"
config.refuse_overwrite(pairs_tsv, meta_json)

kg = kgraph.load(kg_npz)
pharos = kgraph.load(pharos_npz)
print("kg     ", kg.describe(), flush=True)
print("pharos ", pharos.describe(), flush=True)

# both graphs were prepared against the same node table, so node ids are comparable
if not np.array_equal(kg.node_ids[:1000], pharos.node_ids[:1000]):
    raise SystemExit(
        "the two graphs were not prepared against the same node table; re-run kg_prepare.py "
        "for both tags without --nodes"
    )


def pairs_of(graph: kgraph.KnowledgeGraph, predicate: str) -> set[tuple[int, int]]:
    if predicate not in graph.predicates:
        return set()
    rows = graph.edges[graph.edges[:, 1] == graph.predicate_id(predicate)]
    return {(int(s), int(o)) for s, _, o in rows}


kg_pairs, pharos_pairs = pairs_of(kg, args.predicate), pairs_of(pharos, args.predicate)
shared = sorted(kg_pairs & pharos_pairs)
print(
    f"{args.predicate}: {len(kg_pairs):,} pairs in kg, {len(pharos_pairs):,} in pharos, "
    f"{len(shared):,} shared",
    flush=True,
)
if not shared:
    raise SystemExit(
        f"no pair carries {args.predicate} in both graphs, so no identical split exists for "
        f"it. Read {config.OUT / 'kg_census.tsv'} and {config.OUT / 'pharos_census.tsv'} and "
        "pick a predicate both graphs hold."
    )
if len(shared) < 200:
    print(
        f"WARNING: {len(shared)} shared pairs is too few to read a split off. Report the "
        "overlap itself as the result rather than an AUC.",
        flush=True,
    )

# the split: computed once over the shared pairs, so both models see the same rows
splits = {}
for seed in range(args.seeds):
    train, val, test = kgraph.split_rows(
        len(shared), seed, train=args.train_frac, val=args.val_frac
    )
    assignment = np.empty(len(shared), dtype=object)
    assignment[train], assignment[val], assignment[test] = "train", "val", "test"
    splits[seed] = assignment

only_kg = sorted(kg_pairs - pharos_pairs)
only_pharos = sorted(pharos_pairs - kg_pairs)

with open(pairs_tsv, "w") as fh:
    columns = "\t".join(f"split_seed{seed}" for seed in range(args.seeds))
    fh.write(f"subject\tobject\tin_kg\tin_pharos\t{columns}\n")
    for row, (subject, obj) in enumerate(shared):
        assigned = "\t".join(splits[seed][row] for seed in range(args.seeds))
        fh.write(f"{kg.node_ids[subject]}\t{kg.node_ids[obj]}\t1\t1\t{assigned}\n")
    blank = "\t".join("" for _ in range(args.seeds))
    for subject, obj in only_kg:
        fh.write(f"{kg.node_ids[subject]}\t{kg.node_ids[obj]}\t1\t0\t{blank}\n")
    for subject, obj in only_pharos:
        fh.write(f"{kg.node_ids[subject]}\t{kg.node_ids[obj]}\t0\t1\t{blank}\n")

meta = {
    "predicate": args.predicate,
    "seeds": args.seeds,
    "train_frac": args.train_frac,
    "val_frac": args.val_frac,
    "kg_pairs": len(kg_pairs),
    "pharos_pairs": len(pharos_pairs),
    "shared_pairs": len(shared),
    "kg_only_pairs": len(only_kg),
    "pharos_only_pairs": len(only_pharos),
    "jaccard": len(shared) / max(1, len(kg_pairs | pharos_pairs)),
    "n_train": int((splits[0] == "train").sum()),
    "n_val": int((splits[0] == "val").sum()),
    "n_test": int((splits[0] == "test").sum()),
    "note": "the split covers the shared pairs only; both models hold out the same val and "
    "test rows and keep their own remaining edges for message passing",
}
meta_json.write_text(json.dumps(meta, indent=2))

print(f"wrote {pairs_tsv}")
print(f"wrote {meta_json}")
print(json.dumps(meta, indent=2))
