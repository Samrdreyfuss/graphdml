"""Data-generating processes with known effects and known nuisance functions.

Every simulator returns a :class:`~graphdml.datasets.GraphDataset` whose ``extras`` hold the
true nuisances ``m = E[T | X, A]``, ``ell = E[Y | X, A]`` and ``g`` (the outcome
confounding term), so estimators can be run with oracle nuisances (see
:mod:`graphdml.simulate.oracle`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from sklearn.utils import check_random_state

from graphdml.data import GraphData, drop_diagonal
from graphdml.datasets._base import GraphDataset
from graphdml.exposure import row_normalize

__all__ = ["make_linear_gaussian", "make_paper_linear", "make_paper_nonlinear"]


def make_linear_gaussian(
    graph: Any,
    *,
    theta: float = 1.0,
    alpha: float = 0.5,
    n_features: int = 3,
    exposure: str = "mean",
    treatment: str = "continuous",
    noise_t: float = 1.0,
    noise_y: float = 1.0,
    random_state: Any = None,
) -> GraphDataset:
    """Linear model with network confounding, used for exactness and coverage tests.

    ``T = X a + (ĀX) b + e_T`` (or ``Bernoulli(expit(.))`` for ``treatment="binary"``) and
    ``Y = theta T + alpha (E T) + X c + (ĀX) d + e_Y`` with ``Ā`` the mean operator.
    For continuous T and ``exposure="mean"``, ``E[Y | X, A]`` is exactly linear in
    ``NeighborhoodFeatures(aggs=("mean",), hops=2)``, so linear regression is a correctly
    specified nuisance model.
    """
    rng = check_random_state(random_state)
    P = _pattern(graph)
    n, p = P.shape[0], n_features
    M = row_normalize(P)
    E = M if exposure == "mean" else P
    a = np.full(p, 0.5)
    b = np.full(p, 0.7)
    c = np.linspace(1.0, -1.0, p)
    d = np.full(p, 1.0)

    X = rng.standard_normal((n, p))
    MX = M @ X
    index = X @ a + MX @ b
    if treatment == "continuous":
        m = index
        T = m + noise_t * rng.standard_normal(n)
    elif treatment == "binary":
        m = expit(index)
        T = rng.binomial(1, m).astype(float)
    else:
        raise ValueError("treatment must be 'continuous' or 'binary'.")
    g = X @ c + MX @ d
    Y = theta * T + alpha * (E @ T) + g + noise_y * rng.standard_normal(n)
    ell = theta * m + alpha * (E @ m) + g
    data = GraphData(X, T, Y, P)
    return GraphDataset(
        name="linear_gaussian",
        data=data,
        truth={"ade": theta, "ape": alpha},
        exposure=exposure,
        DESCR=make_linear_gaussian.__doc__ or "",
        extras={"m": m, "ell": ell, "g": g},
    )


def make_paper_linear(
    graph: Any,
    *,
    theta: float = 10.0,
    alpha: float = 5.0,
    gamma: float = 0.25,
    noise_sd: float = 0.0,
    random_state: Any = None,
) -> GraphDataset:
    """The linear DGP of Khatami et al. (2025), eq. 38, as in the reference code.

    ``X ~ N(0, 1)``, ``pi = expit((X + gamma A X) / 10)``, ``T ~ Bernoulli(pi)``,
    ``Y = theta T + alpha A T + X + A X (+ noise)`` with ``A`` the binary adjacency and the
    sum exposure. The reference code uses ``gamma = 0.25`` and no outcome noise.
    """
    rng = check_random_state(random_state)
    A = _pattern(graph)
    n = A.shape[0]
    X = rng.standard_normal(n)
    AX = A @ X
    pi = expit((X + gamma * AX) / 10)
    T = rng.binomial(1, pi).astype(float)
    g = X + AX
    Y = theta * T + alpha * (A @ T) + g + noise_sd * rng.standard_normal(n)
    ell = theta * pi + alpha * (A @ pi) + g
    return GraphDataset(
        name="paper_linear",
        data=GraphData(X[:, None], T, Y, A),
        truth={"ade": theta, "ape": alpha},
        exposure="sum",
        DESCR=make_paper_linear.__doc__ or "",
        extras={"m": pi, "ell": ell, "g": g},
    )


def make_paper_nonlinear(
    graph: Any,
    *,
    theta: float = 20.0,
    alpha: float = 5.0,
    gamma: float = 0.25,
    noise_sd: float = 0.0,
    random_state: Any = None,
) -> GraphDataset:
    """The nonlinear DGP of Khatami et al. (2025), eq. 39, with a true neighbor maximum.

    ``pi = expit((X + gamma max_{j in N(i)} X_j) / 10)``,
    ``Y = sigmoid(X + max_{j in N(i)} X_j) + theta T + alpha A T``, implemented as written
    in the paper.
    """
    rng = check_random_state(random_state)
    A = _pattern(graph)
    n = A.shape[0]
    X = rng.standard_normal(n)
    nbr_max = _neighbor_max(A, X)
    pi = expit((X + gamma * nbr_max) / 10)
    T = rng.binomial(1, pi).astype(float)
    g = expit(X + nbr_max)
    Y = theta * T + alpha * (A @ T) + g + noise_sd * rng.standard_normal(n)
    ell = theta * pi + alpha * (A @ pi) + g
    return GraphDataset(
        name="paper_nonlinear",
        data=GraphData(X[:, None], T, Y, A),
        truth={"ade": theta, "ape": alpha},
        exposure="sum",
        DESCR=make_paper_nonlinear.__doc__ or "",
        extras={"m": pi, "ell": ell, "g": g},
    )


def _pattern(graph: Any) -> sp.csr_array:
    A = drop_diagonal(sp.csr_array(graph, dtype=float))
    A.eliminate_zeros()
    A.data = np.ones_like(A.data)
    return A


def _neighbor_max(A: sp.csr_array, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    has = np.diff(A.indptr) > 0
    if has.any():
        out[has] = np.maximum.reduceat(x[A.indices], A.indptr[:-1][has])
    return out
