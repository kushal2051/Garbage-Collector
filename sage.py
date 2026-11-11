"""GraphSAGE and a per-predicate decoder, in plain torch, with neighbour sampling.

No PyG and no DGL. What a convolution needs here is a mean over each node's sampled
neighbours, which is one `index_add_`; a PyG dependency would have to be matched to the
cluster's CUDA build and would buy nothing. If per-predicate convolutions are wanted later,
that is the point to reconsider.

Neighbour sampling, not full-batch, because the merged graph is 1.2M nodes and 12M edges: a
full-batch forward materialises one (2E, hidden) gather per layer, which is ~12 GB per layer at
hidden 128 before autograd stores anything. Sampling bounds it by the batch instead.

What the sampler builds is the *induced* neighbourhood of a batch's endpoints, capped per hop,
and both layers run on that one subgraph. That is not the strict layer-wise block of the
GraphSAGE paper -- a node two hops out is seen by both layers here -- but it is the same
estimator of the same mean aggregation, and it keeps the implementation one function instead of
a block pipeline. Inference uses the same sampler with wider fanouts.

  encoder   two SAGE layers over the sampled TRAINING edges. Message passing never runs over a
            validation or test edge: aggregating over an edge and then predicting it is
            leakage, and it shows up as an AUC near 1.
  decoder   DistMult, score(s, p, o) = <h_s * r_p, h_o>, one learned vector per predicate.
            Multi-relational, which is the minimum for a Biolink graph where the same node pair
            can carry `treats` and `contraindicated_in`.
"""

import numpy as np
import torch
from torch import nn


def _expand(
    indptr: np.ndarray,
    indices: np.ndarray,
    frontier: np.ndarray,
    fanout: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to `fanout` neighbours of every node in `frontier`, vectorised.

    Drawn with replacement, so a node with fewer neighbours than the fanout contributes
    duplicates. That slightly reweights the mean and costs nothing; the alternative is a
    ragged gather per node, which in Python is slower than the training step it feeds.
    """
    frontier = frontier.astype(np.int64)
    degree = indptr[frontier + 1] - indptr[frontier]
    frontier, degree = frontier[degree > 0], degree[degree > 0]
    if not len(frontier):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    take = np.minimum(degree, fanout)
    destination = np.repeat(frontier, take)
    offset = (rng.random(int(take.sum())) * np.repeat(degree, take)).astype(np.int64)
    source = indices[np.repeat(indptr[frontier], take) + offset]
    return source.astype(np.int64), destination


def sample_subgraph(
    indptr: np.ndarray,
    indices: np.ndarray,
    seeds: np.ndarray,
    fanouts: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Induced neighbourhood of `seeds`.

    Returns (nodes, edge_index, seed_positions): the global node ids in the subgraph, the
    message-passing edges relabelled to subgraph positions and made symmetric, and where the
    seeds sit in `nodes`.
    """
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    sources: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    frontier, visited = seeds, seeds
    for fanout in fanouts:
        source, destination = _expand(indptr, indices, frontier, fanout, rng)
        if not len(source):
            break
        sources.append(source)
        destinations.append(destination)
        frontier = np.setdiff1d(np.unique(source), visited, assume_unique=True)
        visited = np.union1d(visited, frontier)

    if not sources:
        nodes = seeds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        source = np.concatenate(sources)
        destination = np.concatenate(destinations)
        nodes = np.union1d(visited, np.unique(source))
        local_source = np.searchsorted(nodes, source)
        local_destination = np.searchsorted(nodes, destination)
        # symmetric: a sampled edge carries a message both ways
        edge_index = torch.stack(
            [
                torch.as_tensor(np.concatenate([local_source, local_destination])),
                torch.as_tensor(np.concatenate([local_destination, local_source])),
            ]
        ).long()
    return nodes, edge_index, torch.as_tensor(np.searchsorted(nodes, seeds)).long()


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of each node's incoming messages; a node with no sampled edge gets zeros."""
    source, target = edge_index
    total = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
    total.index_add_(0, target, x.index_select(0, source))
    count = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    count.index_add_(0, target, torch.ones(target.numel(), device=x.device, dtype=x.dtype))
    return total / count.clamp(min=1.0).unsqueeze(1)


class SAGELayer(nn.Module):
    """h' = W_self h + W_neighbour mean(h_neighbours)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbour_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.self_linear(x) + self.neighbour_linear(mean_aggregate(x, edge_index, x.size(0)))


class LinkPredictor(nn.Module):
    """SAGE encoder + DistMult decoder, over features and an optional node embedding.

    `embed_dim` matters more than any other setting here, and it is the one real modelling
    choice in this stage:

      embed_dim > 0   a learned vector per node, concatenated to its features. This is what
                      ordinary KG link prediction does, and without it the model cannot tell
                      two nodes of the same category apart: mean aggregation over a graph whose
                      input features are a category multi-hot plus a degree gives every node in
                      a category the same representation, and the result sits at chance. Node
                      identity is also exactly what does NOT transfer to an endpoint unseen in
                      training, which is why the seen/unseen breakdown is reported.
      embed_dim = 0   features only: fully inductive, generalises to unseen nodes, and much
                      weaker. Run it to see how much of the score was memorised identity.
    """

    def __init__(
        self,
        in_dim: int,
        num_predicates: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
        num_nodes: int = 0,
        embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = None
        if embed_dim and num_nodes:
            self.embedding = nn.Embedding(num_nodes, embed_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)
            in_dim = in_dim + embed_dim
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(SAGELayer(dims[i], dims[i + 1]) for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.relation = nn.Parameter(torch.randn(num_predicates, hidden) * 0.1)

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor, nodes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`nodes` are the global node ids of `x`'s rows, needed only for the embedding."""
        if self.embedding is not None:
            if nodes is None:
                raise ValueError("a model with an embedding needs the global node ids")
            x = torch.cat([x, self.embedding(nodes)], dim=1)
        h = x
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            h = norm(layer(h, edge_index))
            if index < len(self.layers) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def score(self, h: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        """DistMult logits for (E, 3) triples already relabelled to `h`'s rows."""
        subject, predicate, obj = triples[:, 0], triples[:, 1], triples[:, 2]
        return (
            h.index_select(0, subject)
            * self.relation.index_select(0, predicate)
            * h.index_select(0, obj)
        ).sum(-1)
