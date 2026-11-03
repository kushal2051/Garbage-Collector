"""The RotatE artifacts and the two quantities every stage needs from them.

This exists so detect.py, evaluate.py and the replication cannot drift from one
another: the node layout ([re | im], 256 complex dims in 512 real columns) and
the rotation algebra are written once, here.
"""

import numpy as np
import pandas as pd

import config


class Embeddings:
    """node_embeddings.npy and friends, loaded once (the array is memory-mapped;
    it is 1.9 GB on disk and only the rows we index are read)."""

    def __init__(self):
        config.require(
            config.EMB / "node_embeddings.npy",
            config.EMB / "node_index.tsv",
            config.EMB / "relation_index.tsv",
            config.EMB / "relation_rotations.npy",
        )
        self.nodes = np.load(config.EMB / "node_embeddings.npy", mmap_mode="r")
        self.rots = np.load(config.EMB / "relation_rotations.npy")
        self.D = self.nodes.shape[1] // 2  # complex dims

        idx = pd.read_csv(config.EMB / "node_index.tsv", sep="\t", usecols=["node_id", "curie"])
        self.curies = idx.curie.values
        self.row_of = dict(zip(idx.curie, idx.node_id, strict=False))

        rel = pd.read_csv(config.EMB / "relation_index.tsv", sep="\t")
        self.rel_of = dict(zip(rel.predicate, rel.relation_id, strict=False))

    def relation_id(self, predicate):
        if predicate not in self.rel_of:
            raise SystemExit(f"predicate {predicate!r} not in relation_index.tsv")
        return self.rel_of[predicate]

    def vectors(self, rows):
        return np.asarray(self.nodes[np.asarray(rows, dtype=int)], dtype=np.float64)

    def rotation(self, rid):
        """(cos, sin) of the relation's per-dimension phase."""
        return (
            self.rots[rid, : self.D].astype(np.float64),
            self.rots[rid, self.D :].astype(np.float64),
        )


def map_pairs(edges, predicate, row_of):
    """Rows of `edges` for one predicate, with both endpoints resolved to
    embedding rows. Endpoints absent from the embedding are dropped and counted."""
    e = edges[edges.predicate == predicate].copy()
    e["s_row"] = e.subject.map(row_of)
    e["o_row"] = e.object.map(row_of)
    dropped = int((e.s_row.isna() | e.o_row.isna()).sum())
    e = e.dropna(subset=["s_row", "o_row"]).reset_index(drop=True)
    return e, dropped


def concat_features(emb, e):
    """concat(h, t) -- the original one-class SVM's feature choice: does this
    pair look like the other pairs of this predicate?"""
    h = emb.vectors(e.s_row.values)
    t = emb.vectors(e.o_row.values)
    return np.hstack([h, t])


def rotate_distance(emb, e, rid):
    """||h o r - t||, RotatE's own score for the triple. Not a feature choice --
    a free one-dimensional reference the detector has to beat."""
    h = emb.vectors(e.s_row.values)
    t = emb.vectors(e.o_row.values)
    c, s = emb.rotation(rid)
    D = emb.D
    zre = h[:, :D] * c - h[:, D:] * s - t[:, :D]
    zim = h[:, :D] * s + h[:, D:] * c - t[:, D:]
    mod = np.sqrt(zre**2 + zim**2)
    return np.sqrt((mod**2).sum(1)), mod.sum(1)


def context(e):
    """Node degree and node type for both endpoints, so a flag can be read."""
    deg = pd.read_csv(config.DEGREES_TSV, sep="\t")
    degree = dict(zip(deg.curie, deg.in_degree + deg.out_degree, strict=False))
    types = pd.read_csv(config.NODE_TYPES, sep="\t", usecols=["curie", "selected_type"])
    type_of = dict(zip(types.curie, types.selected_type, strict=False))

    e["subject_type"] = e.subject.map(type_of)
    e["object_type"] = e.object.map(type_of)
    e["subject_degree"] = e.subject.map(degree)
    e["object_degree"] = e.object.map(degree)
    e["min_degree"] = e[["subject_degree", "object_degree"]].min(axis=1)
    return e, degree, type_of
