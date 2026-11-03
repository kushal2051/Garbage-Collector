"""Every external path the pipeline reads, in one place.

Defaults point at the Longleaf locations the original run used. Override any of
them with an environment variable, so the repo runs elsewhere without edits:

    ROTATE_EMB     directory of RotatE artifacts: node_embeddings.npy,
                   node_index.tsv, relation_index.tsv, relation_rotations.npy
    ROTATE_GRAPH   robokop_30fd_filtered directory: subgraph_edges.jsonl,
                   training_node_types.tsv
    ROTATE_DEG_SRC edges.tsv (subject/predicate/object) for the global degree count
    ROTATE_OUT     where stage outputs are written (default: <repo>/out)
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

EMB = Path(
    os.environ.get(
        "ROTATE_EMB", "/work/users/k/k/kkoirala/nvidia-hackathon/rotate_original_embeddings"
    )
)

GRAPH = Path(
    os.environ.get("ROTATE_GRAPH", "/work/users/k/k/kkoirala/n2v2sae/graphs/robokop_30fd_filtered")
)

DEG_SRC = Path(
    os.environ.get(
        "ROTATE_DEG_SRC", "/work/users/k/k/kkoirala/CyclopsHGT/data_set/30fd_robo/edges.tsv"
    )
)

OUT = Path(os.environ.get("ROTATE_OUT", HERE / "out"))

EDGES_JSONL = GRAPH / "subgraph_edges.jsonl"
NODE_TYPES = GRAPH / "training_node_types.tsv"

# derived stage-1/2 products, referenced by every later stage
EDGES_TSV = OUT / "edges.tsv"
DEGREES_TSV = OUT / "degrees.tsv"

# a treats edge is well-formed only if the object is one of these
VALID_TREATS_OBJECT_TYPES = ("biolink:Disease", "biolink:PhenotypicFeature")


def require(*paths):
    """Fail now, with the variable to set, rather than deep inside a stage."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        listed = "\n  ".join(str(m) for m in missing)
        raise SystemExit(
            f"missing input(s):\n  {listed}\n\nSet ROTATE_EMB / ROTATE_GRAPH / "
            "ROTATE_DEG_SRC, or run the earlier stage that produces them."
        )


def refuse_overwrite(*paths):
    """Generated data is read-only once written; stages are re-run under a new
    prefix rather than on top of an existing result."""
    existing = [p for p in paths if Path(p).exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite:\n  {}\n\nPass a different --out-prefix/--tag, "
            "or move the previous result aside.".format("\n  ".join(str(p) for p in existing))
        )
