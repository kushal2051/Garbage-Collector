"""The two things the unsupervised encoder adds, tested without the real graph.

A standalone node table (so a baseline graph keeps the nodes the merged graph lacks), and
the stricter hold-out the encoder needs, since it trains on the edges themselves rather
than on the target predicate alone.
"""

import json

import numpy as np

import kgraph


def write_edges(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_category_for_maps_prefixes_and_falls_back():
    assert kgraph.category_for("NCBIGene:1234") == "biolink:Gene"
    assert kgraph.category_for("PUBCHEM.COMPOUND:1") == "biolink:SmallMolecule"
    assert kgraph.category_for("MONDO:0007254") == "biolink:Disease"
    assert kgraph.category_for("HP:0001250") == "biolink:PhenotypicFeature"
    assert kgraph.category_for("NOTAPREFIX:9") == kgraph.UNKNOWN_CATEGORY


def test_node_table_from_edges_keeps_every_endpoint(tmp_path):
    """The point of the mode: no endpoint is dropped, so nothing dangles."""
    path = write_edges(
        tmp_path / "e.jsonl",
        [
            {"subject": "PUBCHEM.COMPOUND:1", "predicate": "biolink:affects", "object": "NCBIGene:5"},
            {"subject": "NCBIGene:5", "predicate": "biolink:x", "object": "MONDO:9"},
            {"subject": "ODD:7", "predicate": "biolink:x", "object": "MONDO:9"},
        ],
    )
    index, node_ids, category, sets = kgraph.read_nodes_from_edges(path)
    assert set(node_ids) == {"PUBCHEM.COMPOUND:1", "NCBIGene:5", "MONDO:9", "ODD:7"}
    _edges, _preds, _census, dropped = kgraph.read_edges(path, index)
    assert dropped.get("dangling", 0) == 0
    assert sets[category[index["ODD:7"]]] == (kgraph.UNKNOWN_CATEGORY,)


def test_node_table_from_edges_still_dedups_and_drops_self_loops(tmp_path):
    path = write_edges(
        tmp_path / "e.jsonl",
        [
            {"subject": "NCBIGene:5", "predicate": "biolink:x", "object": "MONDO:9"},
            {"subject": "NCBIGene:5", "predicate": "biolink:x", "object": "MONDO:9"},
            {"subject": "HP:3", "predicate": "biolink:x", "object": "HP:3"},
        ],
    )
    index, *_ = kgraph.read_nodes_from_edges(path)
    edges, _preds, _census, dropped = kgraph.read_edges(path, index)
    assert len(edges) == 1
    assert dropped["duplicate_triple"] == 1 and dropped["self_loop"] == 1


def strict_holdout(edges, held_out, num_nodes):
    """The rule gnn.encoder_training_edges applies, isolated so it can be tested alone."""

    def key(subject, obj):
        return subject.astype(np.int64) * num_nodes + obj.astype(np.int64)

    remove = np.concatenate(
        [key(held_out[:, 0], held_out[:, 2]), key(held_out[:, 2], held_out[:, 0])]
    )
    return edges[~np.isin(key(edges[:, 0], edges[:, 2]), remove)]


def test_strict_holdout_removes_the_pair_under_every_predicate_and_both_ways():
    """A held-out pair left in under another predicate trains the encoder on the test edge.

    That leak reads as a high AUC rather than as an error, which is why it is tested.
    """
    num_nodes = 10
    edges = np.array(
        [
            [1, 0, 2],  # the held-out pair, target predicate
            [1, 1, 2],  # SAME pair, different predicate -- must also go
            [2, 1, 1],  # SAME pair reversed -- must also go
            [1, 0, 3],  # a different pair -- must stay
            [4, 2, 5],  # unrelated -- must stay
        ],
        dtype=np.int32,
    )
    held_out = np.array([[1, 0, 2]], dtype=np.int64)
    kept = strict_holdout(edges, held_out, num_nodes)
    assert len(kept) == 2
    assert {(int(s), int(o)) for s, _, o in kept} == {(1, 3), (4, 5)}


def test_strict_holdout_is_stricter_than_the_predicate_only_rule():
    num_nodes = 10
    edges = np.array([[1, 0, 2], [1, 1, 2], [2, 1, 1]], dtype=np.int32)
    held_out = np.array([[1, 0, 2]], dtype=np.int64)
    predicate_only = len(edges) - 1  # what the supervised path would keep
    assert len(strict_holdout(edges, held_out, num_nodes)) < predicate_only
