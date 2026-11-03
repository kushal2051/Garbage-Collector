"""Tests for the rotation algebra and the pair mapping.

No embeddings and no graph: the real artifacts are 1.9 GB and live outside the
repo, so the SUT is exercised against a hand-built stand-in with the same
interface. What is being checked is the arithmetic, which is the part that two
stages have to agree about.
"""

import numpy as np
import pandas as pd
import pytest

import rotate


class FakeEmbeddings:
    """Four nodes in 2 complex dims, laid out [re | im] like the real file."""

    D = 2

    def __init__(self):
        self.nodes = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # node 0: re=(1,0) im=(0,0)
                [0.0, 1.0, 0.0, 0.0],  # node 1
                [0.0, 0.0, 1.0, 0.0],  # node 2
                [2.0, 3.0, 4.0, 5.0],  # node 3
            ]
        )
        # relation 0 is the identity rotation, relation 1 is a quarter turn
        self.rots = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

    def vectors(self, rows):
        return np.asarray(self.nodes[np.asarray(rows, dtype=int)], dtype=np.float64)

    def rotation(self, rid):
        return self.rots[rid, : self.D], self.rots[rid, self.D :]


def pair_frame(s_rows, o_rows):
    return pd.DataFrame({"s_row": s_rows, "o_row": o_rows})


def test_identity_rotation_is_plain_distance():
    """With r = 1 the residual is h - t, so the L2 distance is ||h - t||."""
    emb = FakeEmbeddings()
    e = pair_frame([0, 3], [1, 0])
    l2, l1 = rotate.rotate_distance(emb, e, 0)

    h = emb.vectors([0, 3])
    t = emb.vectors([1, 0])
    assert l2 == pytest.approx(np.linalg.norm(h - t, axis=1))
    # L1 is the sum of the PER-DIMENSION complex moduli, not the vector 1-norm.
    # pair 0: residual re=(1,-1) im=(0,0) -> moduli (1, 1)
    # pair 1: residual re=(1, 3) im=(4,5) -> moduli (sqrt(17), sqrt(34))
    assert l1 == pytest.approx([2.0, np.sqrt(17) + np.sqrt(34)])


def test_quarter_turn_rotates_into_the_imaginary_part():
    """r = i sends (re, im) to (-im, re); a node rotated onto itself is at 0."""
    emb = FakeEmbeddings()
    # node 0 is re=(1,0), im=(0,0); rotating by i gives re=(0,0), im=(1,0) = node 2
    e = pair_frame([0], [2])
    l2, _ = rotate.rotate_distance(emb, e, 1)
    assert l2 == pytest.approx([0.0])


def test_distance_is_zero_for_a_self_consistent_triple():
    emb = FakeEmbeddings()
    e = pair_frame([3], [3])
    l2, l1 = rotate.rotate_distance(emb, e, 0)
    assert l2 == pytest.approx([0.0])
    assert l1 == pytest.approx([0.0])


def test_concat_features_layout():
    """concat(h, t) must be h then t, so the two halves stay interpretable."""
    emb = FakeEmbeddings()
    e = pair_frame([0], [3])
    X = rotate.concat_features(emb, e)
    assert X.shape == (1, 8)
    assert X[0, :4] == pytest.approx(emb.nodes[0])
    assert X[0, 4:] == pytest.approx(emb.nodes[3])


def test_map_pairs_selects_the_predicate_and_drops_unmapped_endpoints():
    edges = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "C"],
            "predicate": ["biolink:treats", "biolink:other", "biolink:treats", "biolink:treats"],
            "object": ["X", "X", "Y", "MISSING"],
        }
    )
    row_of = {"A": 0, "B": 1, "C": 2, "X": 3, "Y": 4}
    e, dropped = rotate.map_pairs(edges, "biolink:treats", row_of)

    assert dropped == 1, "the C -> MISSING edge has an endpoint outside the embedding"
    assert list(e.subject) == ["A", "B"]
    assert list(e.s_row) == [0, 1]
    assert list(e.o_row) == [3, 4]
    assert e.index.tolist() == [0, 1], "index must be reset so positional masks line up"
