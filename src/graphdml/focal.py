"""Focal sets: nodes whose estimating-equation scores are conditionally independent.

Node ``i``'s score depends on the noise of node ``i`` and of every node in the support of
its exposure rows. Call that its *dependency set* ``S_i = {i} ∪ supp(E_i)``. If the
dependency sets of two nodes are disjoint, their scores are independent given ``X``.
A focal set is a maximal collection of nodes with pairwise-disjoint dependency sets. For
a one-hop exposure on an undirected graph this is a set of nodes at pairwise graph
distance >= 3.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state

__all__ = [
    "dependency_matrix",
    "is_independent",
    "is_maximal",
    "select_focal_set",
]


def dependency_matrix(n: int, exposure_matrices: Sequence[sp.csr_array]) -> sp.csr_array:
    """Binary ``(n, n)`` matrix whose row ``i`` is the dependency set ``S_i``."""
    D = sp.csr_array(sp.eye_array(n, format="csr"))
    for E in exposure_matrices:
        pattern = sp.csr_array(E, copy=True)
        pattern.data = np.ones_like(pattern.data)
        D = sp.csr_array(D + pattern)
    D.data = np.ones_like(D.data)
    D.sort_indices()
    return D


def select_focal_set(
    dependency: sp.csr_array,
    strategy: str = "min_degree",
    random_state: int | np.random.RandomState | None = None,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy maximal set of nodes with pairwise-disjoint dependency sets.

    Parameters
    ----------
    strategy : {"min_degree", "random"}
        Order in which nodes are considered. ``"min_degree"`` (ties broken at random)
        yields larger focal sets; ``"random"`` gives a focal set whose degree
        distribution is less skewed.

    candidates : array of node positions, optional
        Only these nodes may be selected (e.g. the population the estimate is about).
        Other nodes can still appear in dependency sets.

    Returns
    -------
    Sorted array of node positions.
    """
    rng = check_random_state(random_state)
    n = dependency.shape[0]
    sizes = np.diff(dependency.indptr)
    if strategy == "min_degree":
        order = np.lexsort((rng.random_sample(n), sizes))
    elif strategy == "random":
        order = rng.permutation(n)
    else:
        raise ValueError(f"Unknown focal-set strategy {strategy!r}.")
    if candidates is not None:
        allowed = np.zeros(n, dtype=bool)
        allowed[np.asarray(candidates)] = True
        order = order[allowed[order]]

    indptr, indices = dependency.indptr, dependency.indices
    used = np.zeros(n, dtype=bool)
    chosen = []
    for i in order:
        cols = indices[indptr[i] : indptr[i + 1]]
        if not used[cols].any():
            used[cols] = True
            chosen.append(i)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def is_independent(dependency: sp.csr_array, focal: np.ndarray) -> bool:
    """True iff the dependency sets of the focal nodes are pairwise disjoint."""
    counts = np.asarray(dependency[np.asarray(focal)].sum(axis=0)).ravel()
    return bool(counts.max(initial=0) <= 1)


def is_maximal(dependency: sp.csr_array, focal: np.ndarray) -> bool:
    """True iff no other node could be added without breaking independence."""
    focal = np.asarray(focal)
    used = np.asarray(dependency[focal].sum(axis=0)).ravel() > 0
    blocked = (dependency @ used.astype(float)) > 0
    others = np.setdiff1d(np.arange(dependency.shape[0]), focal)
    return bool(blocked[others].all())

