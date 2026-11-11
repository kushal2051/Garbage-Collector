"""The graph layout and the split protocol, on a hand-built stand-in.

Like test_rotate.py, these run without the real graph, which is 6.5 GB and lives outside the
repo. Numpy only -- no torch, so they run in the repo's own environment.
"""

import json

import numpy as np
import pytest

import kgraph

NODES = [
    {"id": "NCBIGene:1", "category": ["biolink:Gene", "biolink:BiologicalEntity"]},
    {"id": "NCBIGene:2", "category": ["biolink:Gene", "biolink:BiologicalEntity"]},
    {"id": "MONDO:1", "category": ["biolink:Disease"]},
    {"id": "MONDO:2", "category": ["biolink:Disease"]},
]


def write(tmp_path, nodes, edges):
    (tmp_path / "nodes.jsonl").write_text("".join(json.dumps(n) + "\n" for n in nodes))
    (tmp_path / "edges.jsonl").write_text("".join(json.dumps(e) + "\n" for e in edges))
    return tmp_path / "nodes.jsonl", tmp_path / "edges.jsonl"


def test_loader_dedups_triples_and_drops_loops_and_dangling(tmp_path):
    edges = [
        {"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "MONDO:1"},
        # the merged graph has 22 sources; the same triple twice must stay one triple, or the
        # duplicate lands in train and in test and the model scores it from memory
        {"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "MONDO:1"},
        {"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "NCBIGene:1"},
        {"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "ABSENT:9"},
        {"subject": "NCBIGene:1", "object": "MONDO:1"},
    ]
    graph = kgraph.load_jsonl(*write(tmp_path, NODES, edges))
    assert graph.num_nodes == 4
    assert graph.num_edges == 1
    assert graph.dropped == {
        "self_loop": 1,
        "dangling": 1,
        "malformed": 1,
        "duplicate_triple": 1,
    }


def test_category_sets_are_shared_not_repeated_per_node(tmp_path):
    """ORION writes every ancestor, so the lists are long and highly repeated; one int per
    node is what keeps the node table small."""
    edges = [{"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "MONDO:1"}]
    graph = kgraph.load_jsonl(*write(tmp_path, NODES, edges))
    assert len(graph.category_sets) == 2, "two distinct lists across four nodes"
    assert graph.category[0] == graph.category[1]
    assert graph.category[2] == graph.category[3]


def test_save_and_load_round_trip(tmp_path):
    edges = [{"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "MONDO:1"}]
    graph = kgraph.load_jsonl(*write(tmp_path, NODES, edges))
    kgraph.save(tmp_path / "g.npz", graph)
    back = kgraph.load(tmp_path / "g.npz")
    assert back.node_ids.tolist() == graph.node_ids.tolist()
    assert back.category_sets == graph.category_sets
    assert back.predicates == graph.predicates
    assert np.array_equal(back.edges, graph.edges)


def test_predicate_id_names_what_is_present(tmp_path):
    edges = [{"subject": "NCBIGene:1", "predicate": "biolink:affects", "object": "MONDO:1"}]
    graph = kgraph.load_jsonl(*write(tmp_path, NODES, edges))
    assert graph.predicate_id("biolink:affects") == 0
    with pytest.raises(SystemExit, match="biolink:treats is not in this graph"):
        graph.predicate_id("biolink:treats")


# ------------------------------------------------------------------ adjacency and features


def test_csr_lists_both_directions_of_every_edge():
    edges = np.array([[0, 0, 2], [1, 0, 2]], dtype=np.int32)
    indptr, indices = kgraph.csr(edges, num_nodes=3)
    assert indices[indptr[0] : indptr[1]].tolist() == [2]
    assert indices[indptr[1] : indptr[2]].tolist() == [2]
    assert sorted(indices[indptr[2] : indptr[3]].tolist()) == [0, 1]


def test_features_use_training_degree_only():
    category = np.array([0, 0, 1, 1], dtype=np.int32)
    sets = [("biolink:Gene",), ("biolink:Disease",)]
    all_edges = np.array([[0, 0, 2], [1, 0, 3]], dtype=np.int32)
    train = all_edges[:1]

    x = kgraph.features(category, sets, train, num_nodes=4)
    train_degree = kgraph.degrees(train, 4)
    assert x.dtype == np.float32
    assert np.allclose(x[:, -2], np.log1p(train_degree))
    assert np.array_equal(x[:, -1] == 1.0, train_degree == 0)
    assert kgraph.degrees(train, 4).sum() < kgraph.degrees(all_edges, 4).sum()
    # the category columns are the multi-hot, one column per kept category
    assert x[0, 0] != x[2, 0], "a gene and a disease must not share a feature row"


# ------------------------------------------------------------------ the split protocol


def test_split_rows_are_disjoint_and_cover_everything():
    train, val, test = kgraph.split_rows(1000, seed=0)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    assert set(train) | set(val) | set(test) == set(range(1000))
    assert not set(train) & set(val) and not set(train) & set(test)
    assert not set(val) & set(test)


def test_split_rows_differ_by_seed():
    assert set(kgraph.split_rows(1000, 0)[0]) != set(kgraph.split_rows(1000, 1)[0])


def test_split_rows_rejects_fractions_that_leave_no_test():
    with pytest.raises(ValueError, match="leaves no test rows"):
        kgraph.split_rows(100, seed=0, train=0.95, val=0.1)


def test_negatives_are_filtered_and_type_matched():
    genes, diseases = np.array([0, 1, 2]), np.array([10, 11, 12])
    positives = np.array([[0, 0, 10], [1, 0, 11]], dtype=np.int32)
    known = {(0, 0, 10), (1, 0, 11)}
    negatives = kgraph.sample_negatives(positives, known, genes, diseases, seed=0)
    assert len(negatives) == 2
    for subject, predicate, obj in negatives:
        assert (int(subject), int(predicate), int(obj)) not in known
        assert subject != obj
        # one endpoint is kept, the corrupted one comes from the matching pool
        assert int(subject) in set(genes.tolist()) and int(obj) in set(diseases.tolist())


def test_side_pools_are_the_nodes_on_each_side_of_that_predicate():
    edges = np.array([[0, 0, 10], [1, 0, 11], [5, 1, 6]], dtype=np.int32)
    subjects, objects = kgraph.side_pools(edges, predicate_id=0)
    assert subjects.tolist() == [0, 1]
    assert objects.tolist() == [10, 11]


def test_both_seen_marks_unseen_endpoints():
    edges = np.array([[0, 0, 1], [2, 0, 3]], dtype=np.int32)
    assert kgraph.both_seen(edges, edges[:1]).tolist() == [True, False]


def test_both_seen_handles_a_node_id_above_the_training_range():
    """A test edge on a node larger than anything in training must not index out of bounds."""
    train = np.array([[0, 0, 1]], dtype=np.int32)
    test = np.array([[0, 0, 99]], dtype=np.int32)
    assert kgraph.both_seen(test, train).tolist() == [False]


def test_degree_scores_rank_well_connected_pairs_higher():
    train = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.int32)
    scores = kgraph.degree_scores(np.array([[0, 0, 1], [2, 0, 3]], dtype=np.int32), train, 4)
    assert scores[0] > scores[1]
