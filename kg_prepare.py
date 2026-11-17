"""Stage 5a. Parse a jsonl graph once into compact arrays every later stage reads.

The merged graph is 6.5 GB of jsonl and takes minutes to parse; `pharos_split.py` and `gnn.py`
read `out/<tag>.npz` instead. Like stage 1, the pass takes a full predicate census while it is
reading anyway, so choosing a different predicate later is a one-flag change and not a re-read.

Two ways to map a graph's endpoints to node ids:

  shared node table (default)   both graphs mapped through the merged graph's `nodes.jsonl`.
                                One node id space, so a split can be identical by construction.
                                An endpoint absent from that table is counted as `dangling`.
  --nodes-from-edges            the graph gets its OWN node table, built from the endpoints its
                                edge file mentions, with categories from the CURIE prefix.

The default silently costs a standalone baseline most of its graph. Measured on Pharos v319
normalized against the merged table: 9.1% of its 307,917 nodes survive, 64.7% of its edges are
dangling, and the loss is almost entirely compounds -- 207 of 259,697 PUBCHEM nodes are in the
merged table, which carries 406 in total. The drug-target edges that make Pharos worth using as
a baseline cannot survive that join. Use `--nodes-from-edges` when the graph is being encoded on
its own, and let `pharos_split.py` join the two on CURIEs instead of on node ids.

Usage:
    python kg_prepare.py --tag kg        # the KGB-CIA merged graph
    python kg_prepare.py --tag pharos    # the Pharos baseline, same node table
    python kg_prepare.py --tag pharos319 --nodes-from-edges \
        --edges /path/to/pharos_v319/pharos_edges_normalized.jsonl
"""

import argparse
import json

import config
import kgraph

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, help="output name: out/<tag>.npz")
ap.add_argument("--nodes", default=None, help="override the node table (default: the merged graph)")
ap.add_argument("--edges", default=None, help="override the edges jsonl for this tag")
ap.add_argument(
    "--nodes-from-edges",
    action="store_true",
    help="build the node table from the edge file's own endpoints instead of a nodes.jsonl, "
    "with categories derived from the CURIE prefix. Use for a standalone graph that is NOT "
    "being intersected with the merged graph -- it keeps every node the file mentions.",
)
args = ap.parse_args()

DEFAULT_EDGES = {"kg": config.KG_EDGES, "pharos": config.PHAROS}
edges_path = args.edges or DEFAULT_EDGES.get(args.tag)
if edges_path is None:
    raise SystemExit(f"--tag {args.tag} has no default edge file; pass --edges")
nodes_path = None if args.nodes_from_edges else (args.nodes or config.KG_NODES)
if args.nodes_from_edges and args.nodes:
    raise SystemExit("--nodes-from-edges and --nodes are mutually exclusive")

config.require(*(p for p in (nodes_path, edges_path) if p is not None))
config.OUT.mkdir(parents=True, exist_ok=True)
npz = config.OUT / f"{args.tag}.npz"
census_tsv = config.OUT / f"{args.tag}_census.tsv"
config.refuse_overwrite(npz, npz.with_suffix(".meta.json"), census_tsv)

if args.nodes_from_edges:
    print(f"node table from the edges themselves: {edges_path}", flush=True)
    index, node_ids, category, category_sets = kgraph.read_nodes_from_edges(edges_path)
else:
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
