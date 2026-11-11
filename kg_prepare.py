"""Stage 5a. Parse a jsonl graph once into compact arrays every later stage reads.

The merged graph is 6.5 GB of jsonl and takes minutes to parse; `pharos_split.py` and `gnn.py`
read `out/<tag>.npz` instead. Like stage 1, the pass takes a full predicate census while it is
reading anyway, so choosing a different predicate later is a one-flag change and not a re-read.

Both graphs are mapped through the SAME node table -- the merged graph's `nodes.jsonl`. That is
what makes the comparison possible at all: an identical split needs one node id space, and it
is also where Pharos's node categories come from, since `pharos_edges.jsonl` ships no nodes.
A Pharos edge whose endpoint is absent from that table is counted as `dangling`, not dropped
silently.

Usage:
    python kg_prepare.py --tag kg        # the KGB-CIA merged graph
    python kg_prepare.py --tag pharos    # the Pharos baseline, same node table
"""

import argparse
import json

import config
import kgraph

ap = argparse.ArgumentParser()
ap.add_argument("--tag", choices=("kg", "pharos"), required=True)
ap.add_argument("--nodes", default=None, help="override the node table (default: the merged graph)")
ap.add_argument("--edges", default=None, help="override the edges jsonl for this tag")
args = ap.parse_args()

nodes_path = args.nodes or config.KG_NODES
edges_path = args.edges or (config.KG_EDGES if args.tag == "kg" else config.PHAROS)

config.require(nodes_path, edges_path)
config.OUT.mkdir(parents=True, exist_ok=True)
npz = config.OUT / f"{args.tag}.npz"
census_tsv = config.OUT / f"{args.tag}_census.tsv"
config.refuse_overwrite(npz, npz.with_suffix(".meta.json"), census_tsv)

print(f"node table {nodes_path}", flush=True)
index, node_ids, category, category_sets = kgraph.read_nodes(nodes_path)
print(f"  {len(node_ids):,} nodes, {len(category_sets):,} distinct category lists", flush=True)

print(f"edges {edges_path}", flush=True)
edges, predicates, census, dropped = kgraph.read_edges(edges_path, index)
kg = kgraph.KnowledgeGraph(node_ids, category, category_sets, predicates, edges, dict(dropped))
print(" ", kg.describe(), flush=True)

kgraph.save(npz, kg)
with open(census_tsv, "w") as fh:
    fh.write("predicate\tedges\n")
    for predicate, count in census.most_common():
        fh.write(f"{predicate}\t{count}\n")

print(f"wrote {npz}")
print(f"wrote {census_tsv}")
print("census:", json.dumps(dict(census.most_common(10)), indent=2))
