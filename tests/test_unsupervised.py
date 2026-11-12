"""The two things the unsupervised encoder adds, tested without the real graph.

A standalone node table, so a baseline graph keeps the nodes the merged graph lacks.
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
