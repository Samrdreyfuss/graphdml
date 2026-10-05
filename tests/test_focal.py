import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path

from graphdml.exposure import MeanExposure, TwoHopExposure
from graphdml.focal import dependency_matrix, is_independent, is_maximal, select_focal_set
from graphdml.simulate import barabasi_albert, erdos_renyi, stochastic_block_model

GRAPHS = {
    "er": lambda s: erdos_renyi(400, 4.0, random_state=s),
    "ba": lambda s: barabasi_albert(400, 2, random_state=s),
    "sbm": lambda s: stochastic_block_model([40] * 10, 0.2, 0.002, random_state=s),
}


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("strategy", ["min_degree", "random"])
@pytest.mark.parametrize("seed", range(3))
def test_one_hop_focal_set_is_distance_three_and_maximal(graph, strategy, seed):
    A = GRAPHS[graph](seed)
    dep = dependency_matrix(A.shape[0], [A])
    focal = select_focal_set(dep, strategy, random_state=seed)
    assert is_independent(dep, focal)
    assert is_maximal(dep, focal)
    dist = shortest_path(A, unweighted=True, indices=focal)[:, focal]
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 3


def test_two_hop_dependency_requires_distance_five():
    A = erdos_renyi(300, 3.0, random_state=0)
    from graphdml import GraphData

    data = GraphData(np.zeros((300, 1)), np.zeros(300), np.zeros(300), A)
    E = [MeanExposure().matrix(data), TwoHopExposure().matrix(data)]
    dep = dependency_matrix(300, E)
    focal = select_focal_set(dep, random_state=0)
    dist = shortest_path(A, unweighted=True, indices=focal)[:, focal]
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 5
    assert is_maximal(dep, focal)


def test_detects_invalid_sets():
    A = erdos_renyi(100, 4.0, random_state=0)
    dep = dependency_matrix(100, [A])
    i, j = A.nonzero()
    assert not is_independent(dep, np.array([i[0], j[0]]))
    assert not is_maximal(dep, np.array([], dtype=int))


def test_min_degree_strategy_finds_larger_sets_on_average():
    sizes = {"min_degree": [], "random": []}
    for seed in range(5):
        A = barabasi_albert(500, 2, random_state=seed)
        dep = dependency_matrix(500, [A])
        for s in sizes:
            sizes[s].append(len(select_focal_set(dep, s, random_state=seed)))
    assert np.mean(sizes["min_degree"]) > np.mean(sizes["random"])


def test_candidates_restrict_selection():
    A = erdos_renyi(400, 4.0, random_state=0)
    dep = dependency_matrix(400, [A])
    cand = np.arange(0, 400, 2)
    focal = select_focal_set(dep, random_state=0, candidates=cand)
    assert set(focal) <= set(cand) and is_independent(dep, focal)
