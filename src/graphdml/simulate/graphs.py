"""Random graph generators returning symmetric binary adjacency matrices (numpy/scipy only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.utils import check_random_state

from graphdml.data import _adjacency_from_edges

__all__ = [
    "barabasi_albert",
    "edges_to_adjacency",
    "erdos_renyi",
    "geometric_graph",
    "random_geometric",
    "stochastic_block_model",
]


def edges_to_adjacency(n: int, edges: Any) -> sp.csr_array:
    """Symmetric binary adjacency from undirected ``(i, j)`` pairs (duplicates ignored)."""
    return _adjacency_from_edges(n, edges, directed=False)


def erdos_renyi(n: int, avg_degree: float, random_state: Any = None) -> sp.csr_array:
    """G(n, p) with ``p = avg_degree / (n - 1)``."""
    rng = check_random_state(random_state)
    p = avg_degree / (n - 1)
    m = rng.binomial(n * (n - 1) // 2, p)
    i, j = rng.randint(n, size=m), rng.randint(n, size=m)
    keep = i != j
    return edges_to_adjacency(n, np.column_stack([i[keep], j[keep]]))


def barabasi_albert(n: int, m: int, random_state: Any = None) -> sp.csr_array:
    """Preferential attachment: each new node links to ``m`` existing nodes (hubs emerge)."""
    rng = check_random_state(random_state)
    if not 1 <= m < n:
        raise ValueError("Need 1 <= m < n.")
    edges: list[tuple[int, int]] = []
    repeated: list[int] = []
    targets = list(range(m))
    for v in range(m, n):
        edges.extend((v, t) for t in targets)
        repeated.extend(targets)
        repeated.extend([v] * m)
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.randint(len(repeated))])
        targets = list(chosen)
    return edges_to_adjacency(n, np.asarray(edges))


def stochastic_block_model(
    sizes: Any,
    p_in: float,
    p_out: float,
    random_state: Any = None,
    weights: Any = None,
) -> sp.csr_array:
    """Communities with dense within-block and sparse between-block edges.

    ``weights`` (one per node, mean ~1) scale within-block edge probabilities as
    ``p_in * w_i * w_j``, so some nodes are more sociable than others.
    """
    rng = check_random_state(random_state)
    sizes = np.asarray(sizes, dtype=int)
    n = int(sizes.sum())
    block = np.repeat(np.arange(len(sizes)), sizes)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    parts = []
    start = 0
    for s in sizes:
        iu, ju = np.triu_indices(s, 1)
        p = np.minimum(1.0, p_in * w[start + iu] * w[start + ju])
        keep = rng.random_sample(len(iu)) < p
        parts.append(np.column_stack([start + iu[keep], start + ju[keep]]))
        start += s
    n_between = (n * n - int((sizes**2).sum())) // 2
    m = rng.binomial(n_between, p_out) if n_between > 0 else 0
    i, j = rng.randint(n, size=m), rng.randint(n, size=m)
    keep = block[i] != block[j]
    parts.append(np.column_stack([i[keep], j[keep]]))
    return edges_to_adjacency(n, np.vstack(parts))


def geometric_graph(coords: Any, avg_degree: float) -> sp.csr_array:
    """Connect points within a radius, calibrated so the mean degree is ~``avg_degree``."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    tree = cKDTree(coords)
    span = np.prod(np.ptp(coords, axis=0)) or 1.0
    r = (avg_degree * span / (np.pi * n)) ** (1 / coords.shape[1])
    for _ in range(4):
        pairs = tree.query_pairs(r, output_type="ndarray")
        realised = 2 * len(pairs) / n
        if realised == 0:
            r *= 2
            continue
        if abs(realised - avg_degree) / avg_degree < 0.05:
            break
        r *= (avg_degree / realised) ** (1 / coords.shape[1])
    pairs = tree.query_pairs(r, output_type="ndarray")
    return edges_to_adjacency(n, pairs)


def random_geometric(
    n: int, avg_degree: float, random_state: Any = None, return_positions: bool = False
) -> Any:
    """Points in the unit square linked to nearby points (e.g. neighbors in a town)."""
    rng = check_random_state(random_state)
    pos = rng.random_sample((n, 2))
    A = geometric_graph(pos, avg_degree)
    return (A, pos) if return_positions else A
