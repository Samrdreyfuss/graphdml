import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from graphdml import GraphData, GraphDMLWarning


def _xty(n):
    return np.zeros((n, 1)), np.zeros(n), np.zeros(n)


def test_undirected_edges_are_symmetric():
    data = GraphData.from_edges(*_xty(3), [(0, 1), (1, 2)])
    A = data.adjacency.toarray()
    assert (A == A.T).all()
    assert data.n_edges == 2
    assert data.degree.tolist() == [1, 2, 1]


def test_directed_convention_source_influences_target():
    data = GraphData.from_edges(*_xty(2), [(0, 1)], directed=True)
    # node 1 is influenced by node 0  =>  adjacency[1, 0] == 1
    assert data.adjacency[1, 0] == 1
    assert data.adjacency[0, 1] == 0
    assert data.degree.tolist() == [0, 1]


def test_duplicates_and_self_loops_are_dropped():
    with pytest.warns(GraphDMLWarning, match="self-loop"):
        data = GraphData.from_edges(*_xty(3), [(0, 1), (1, 0), (0, 1), (2, 2)])
    assert data.n_edges == 1
    assert data.adjacency.diagonal().sum() == 0


def test_undirected_duplicate_weights_stay_symmetric():
    data = GraphData.from_edges(*_xty(2), [(0, 1), (1, 0)], weights=[1.0, 2.0])
    A = data.adjacency.toarray()
    assert A[0, 1] == A[1, 0] == 1.0


def test_validation_errors():
    X, T, Y = _xty(3)
    with pytest.raises(ValueError, match="NaN"):
        GraphData.from_edges(np.array([[np.nan], [0], [0]]), T, Y, [])
    with pytest.raises(ValueError, match="entries"):
        GraphData.from_edges(X, np.zeros(2), Y, [])
    with pytest.raises(ValueError, match="symmetric"):
        GraphData(X, T, Y, sp.csr_array(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])))
    with pytest.raises(ValueError, match="node positions"):
        GraphData.from_edges(X, T, Y, [(0, 5)])


def test_from_pandas_maps_ids_and_roundtrips():
    nodes = pd.DataFrame(
        {"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "t": [0, 1, 0], "y": [0.5, 1.5, 2.5]}
    )
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    data = GraphData.from_pandas(
        nodes, edges, features=["x"], treatment="t", outcome="y", node_id="id"
    )
    assert data.node_ids.tolist() == ["a", "b", "c"]
    assert data.adjacency[0, 1] == data.adjacency[1, 0] == 1
    n_df, e_df = data.to_pandas()
    assert n_df["x"].tolist() == [1.0, 2.0, 3.0]
    assert sorted(map(tuple, e_df[["source", "target"]].to_numpy().tolist())) == [
        ("a", "b"),
        ("b", "c"),
    ]


def test_from_pandas_unknown_ids():
    nodes = pd.DataFrame({"x": [1.0, 2.0], "t": [0, 1], "y": [0.0, 1.0]})
    edges = pd.DataFrame({"source": [0], "target": [7]})
    with pytest.raises(ValueError, match="unknown node ids"):
        GraphData.from_pandas(nodes, edges, features=["x"], treatment="t", outcome="y")


def test_from_networkx():
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    for i in range(3):
        G.add_node(i, x=float(i), t=i % 2, y=float(i))
    G.add_edges_from([(0, 1), (1, 2)])
    data = GraphData.from_networkx(G, features=["x"], treatment="t", outcome="y")
    assert data.n_edges == 2 and not data.directed


def test_treatment_type_and_with_values():
    data = GraphData.from_edges(np.zeros((3, 1)), [0, 1, 1], [0, 0, 0], [])
    assert data.treatment_is_binary
    data2 = data.with_values(T=[0.5, 1.0, 2.0])
    assert not data2.treatment_is_binary
    assert data.treatment_is_binary  # original unchanged
