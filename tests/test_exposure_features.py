import numpy as np
import pytest

from graphdml import (
    GraphData,
    MeanExposure,
    NeighborhoodFeatures,
    SumExposure,
    TwoHopExposure,
)
from graphdml.exposure import resolve_exposures, row_normalize


def path_graph(n, T=None, X=None, directed=False):
    T = np.arange(n, dtype=float) if T is None else T
    X = np.arange(n, dtype=float)[:, None] if X is None else X
    return GraphData.from_edges(X, T, np.zeros(n), [(i, i + 1) for i in range(n - 1)],
                                directed=directed)


def test_sum_and_mean_exposure():
    data = path_graph(3, T=np.array([1.0, 0.0, 1.0]))
    assert SumExposure()(data).tolist() == [0.0, 2.0, 0.0]
    assert MeanExposure()(data).tolist() == [0.0, 1.0, 0.0]


def test_isolated_nodes_have_zero_exposure():
    data = GraphData.from_edges(np.zeros((3, 1)), [1, 1, 1], [0, 0, 0], [(0, 1)])
    assert SumExposure()(data)[2] == 0
    assert MeanExposure()(data)[2] == 0


def test_directed_exposure_uses_in_neighbors():
    data = path_graph(2, T=np.array([1.0, 0.0]), directed=True)  # 0 -> 1
    assert SumExposure()(data).tolist() == [0.0, 1.0]


def test_two_hop_excludes_self_and_direct_neighbors():
    # triangle 0-1-2 plus tail 2-3: node 0's two-hop set is {3} only
    data = GraphData.from_edges(
        np.zeros((4, 1)), [0, 0, 0, 1], [0, 0, 0, 0], [(0, 1), (1, 2), (0, 2), (2, 3)]
    )
    M = TwoHopExposure().matrix(data).toarray()
    assert M[0].tolist() == [0, 0, 0, 1]
    assert M[3].tolist() == [1, 1, 0, 0]
    assert np.diag(M).sum() == 0


def test_resolve_exposures():
    assert resolve_exposures(None) == []
    assert [e.name for e in resolve_exposures(["sum", "two_hop_sum"])] == ["sum", "two_hop_sum"]
    with pytest.raises(ValueError, match="Unknown exposure"):
        resolve_exposures("nope")
    with pytest.raises(ValueError, match="unique"):
        resolve_exposures(["sum", "sum"])


def test_neighborhood_aggregates_handle_negatives_and_isolates():
    X = np.array([[-5.0], [-1.0], [-3.0], [7.0]])
    data = GraphData.from_edges(X, np.zeros(4), np.zeros(4), [(0, 1), (0, 2)])
    F = NeighborhoodFeatures(aggs=("mean", "max", "min", "sum"), hops=1).transform(data)
    names = NeighborhoodFeatures(aggs=("mean", "max", "min", "sum"), hops=1).get_feature_names_out(
        data
    )
    assert F.shape == (4, len(names))
    col = {n: F[:, j] for j, n in enumerate(names)}
    assert col["mean_nbr(X0)"].tolist() == [-2.0, -5.0, -5.0, 0.0]
    assert col["max_nbr(X0)"].tolist() == [-1.0, -5.0, -5.0, 0.0]  # not 0 for node 0
    assert col["min_nbr(X0)"].tolist() == [-3.0, -5.0, -5.0, 0.0]
    assert col["sum_nbr(X0)"].tolist() == [-4.0, -5.0, -5.0, 0.0]
    assert col["degree"].tolist() == [2, 1, 1, 0]


def test_two_hops_is_mean_propagation():
    rng = np.random.RandomState(0)
    data = path_graph(6, X=rng.standard_normal((6, 2)))
    F = NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False).transform(data)
    M = row_normalize(data.pattern)
    expected = np.hstack([data.X, M @ data.X, M @ (M @ data.X)])
    np.testing.assert_allclose(F, expected)
