"""The sampler and the model. Skipped where torch is absent: it is not a repo dependency.

torch lives in the GPU environment the Slurm job uses (slurm/env.sh, PYTHON_GNN), not in the
repo's uv env, so these tests skip rather than fail where it is missing -- the same reason
stage 4b's umap dependency is a separate group.
"""

import importlib.util

import numpy as np
import pytest

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed in this env"
)

# a 4-node path 0-1-2-3, as CSR
EDGES = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=np.int32)


def csr():
    import kgraph

    return kgraph.csr(EDGES, num_nodes=4)


def test_expand_respects_the_fanout_and_skips_isolated_nodes():
    import sage

    indptr, indices = csr()
    rng = np.random.default_rng(0)
    source, destination = sage._expand(indptr, indices, np.array([1]), fanout=1, rng=rng)
    assert len(source) == 1
    assert destination.tolist() == [1]
    assert source[0] in (0, 2)

    isolated_indptr, isolated_indices = np.array([0, 0]), np.zeros(0, dtype=np.int64)
    source, _ = sage._expand(isolated_indptr, isolated_indices, np.array([0]), 5, rng)
    assert len(source) == 0


def test_expand_never_returns_more_than_the_degree():
    import sage

    indptr, indices = csr()
    rng = np.random.default_rng(0)
    # node 0 has one neighbour; a fanout of 10 must not invent nine more
    source, _ = sage._expand(indptr, indices, np.array([0]), fanout=10, rng=rng)
    assert len(source) == 1


@requires_torch
def test_sample_subgraph_relabels_and_keeps_the_seeds():
    import sage

    indptr, indices = csr()
    rng = np.random.default_rng(0)
    nodes, edge_index, seed_positions = sage.sample_subgraph(
        indptr, indices, np.array([0]), fanouts=(10, 10), rng=rng
    )
    assert 0 in nodes.tolist()
    assert nodes.tolist() == sorted(nodes.tolist()), "nodes must be sorted for searchsorted"
    assert nodes[seed_positions[0]] == 0
    assert int(edge_index.max()) < len(nodes), "edges must be relabelled into the subgraph"
    assert edge_index.shape[1] % 2 == 0, "every sampled edge carries a message both ways"


@requires_torch
def test_sample_subgraph_of_an_isolated_seed_is_the_seed_alone():
    import sage

    rng = np.random.default_rng(0)
    nodes, edge_index, seeds = sage.sample_subgraph(
        np.array([0, 0]), np.zeros(0, dtype=np.int64), np.array([0]), (5,), rng
    )
    assert nodes.tolist() == [0]
    assert edge_index.shape == (2, 0)
    assert seeds.tolist() == [0]


@requires_torch
def test_mean_aggregate_averages_neighbours_and_zeroes_isolated_nodes():
    import torch

    import sage

    x = torch.tensor([[1.0], [3.0], [5.0]])
    edge_index = torch.tensor([[1, 2], [0, 0]])  # messages into node 0 only
    out = sage.mean_aggregate(x, edge_index, num_nodes=3)
    assert out[0].item() == pytest.approx(4.0)
    assert out[2].item() == 0.0


@requires_torch
def test_an_embedding_model_refuses_to_encode_without_node_ids():
    """The embedding is a lookup by global node id; silently skipping it would score garbage."""
    import torch

    import sage

    model = sage.LinkPredictor(in_dim=4, num_predicates=1, num_nodes=10, embed_dim=8)
    with pytest.raises(ValueError, match="global node ids"):
        model.encode(torch.randn(3, 4), torch.tensor([[0], [1]]))


@requires_torch
def test_encoder_and_decoder_shapes():
    import torch

    import sage

    model = sage.LinkPredictor(in_dim=4, num_predicates=3, hidden=8, layers=2)
    h = model.encode(torch.randn(5, 4), torch.tensor([[0, 1], [1, 0]]))
    assert h.shape == (5, 8)
    assert model.score(h, torch.tensor([[0, 0, 1], [2, 1, 3]])).shape == (2,)


@requires_torch
def test_the_predicate_changes_the_score():
    """A node pair can carry treats and contraindicated_in; the predicate has to matter."""
    import torch

    import sage

    torch.manual_seed(0)
    model = sage.LinkPredictor(in_dim=4, num_predicates=2, hidden=8, layers=1)
    h = torch.randn(3, 8)
    a, b = model.score(h, torch.tensor([[0, 0, 1], [0, 1, 1]])).tolist()
    assert a != pytest.approx(b)


@requires_torch
def test_the_model_learns_structure_a_degree_baseline_cannot_see():
    """The smoke test that matters: two communities, no category signal, degree at chance.

    Node features carry only degree here, so a model that learns nothing from its neighbours
    scores like the baseline. A clear gap means the sampler, the encoder and the decoder are
    wired together correctly.
    """
    import torch

    import kgraph
    import sage

    rng = np.random.default_rng(0)
    # two blocks of 60; edges within a block only
    blocks = [np.arange(0, 60), np.arange(60, 120)]
    rows = set()
    while len(rows) < 1200:
        block = blocks[rng.integers(2)]
        a, b = rng.choice(block, size=2, replace=False)
        if a != b:
            rows.add((int(min(a, b)), 0, int(max(a, b))))
    edges = np.array(sorted(rows), dtype=np.int32)

    train_i, _, test_i = kgraph.split_rows(len(edges), seed=0)
    train, test = edges[train_i], edges[test_i]
    pool = np.arange(120)
    known = {(int(s), 0, int(o)) for s, _, o in edges}
    negatives = kgraph.sample_negatives(test, known, pool, pool, seed=0)

    x = np.column_stack(
        [np.log1p(kgraph.degrees(train, 120)), np.ones(120, dtype=np.float32)]
    ).astype(np.float32)
    indptr, indices = kgraph.csr(train, 120)

    torch.manual_seed(0)
    model = sage.LinkPredictor(
        in_dim=2,
        num_predicates=1,
        hidden=32,
        layers=2,
        dropout=0.0,
        num_nodes=120,
        embed_dim=16,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(triples):
        seeds = np.unique(np.concatenate([triples[:, 0], triples[:, 2]]))
        nodes, edge_index, _ = sage.sample_subgraph(indptr, indices, seeds, (15, 10), rng)
        h = model.encode(torch.as_tensor(x[nodes]), edge_index, torch.as_tensor(nodes).long())
        local = np.column_stack(
            [
                np.searchsorted(nodes, triples[:, 0]),
                triples[:, 1],
                np.searchsorted(nodes, triples[:, 2]),
            ]
        )
        return model.score(h, torch.as_tensor(local).long())

    for epoch in range(120):
        train_negatives = kgraph.sample_negatives(train, known, pool, pool, seed=epoch)
        triples = np.concatenate([train, train_negatives])
        labels = torch.cat([torch.ones(len(train)), torch.zeros(len(train_negatives))])
        optimiser.zero_grad()
        loss_fn(forward(triples), labels).backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        positive = forward(test).numpy()
        negative = forward(negatives).numpy()

    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    gnn = roc_auc_score(labels, np.concatenate([positive, negative]))
    degree = roc_auc_score(
        labels,
        np.concatenate(
            [
                kgraph.degree_scores(test, train, 120),
                kgraph.degree_scores(negatives, train, 120),
            ]
        ),
    )
    assert gnn > degree + 0.05, f"gnn {gnn:.3f} is not clear of degree {degree:.3f}"
