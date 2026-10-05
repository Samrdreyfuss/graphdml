"""Exposure maps: how neighbors' treatments are summarised into a node's peer exposure.

GraphDML supports *linear* exposure maps ``Z = E @ T`` for a sparse ``(n, n)`` operator
``E``. Linearity is what lets the peer residual be computed as ``E @ (T - m_hat)``
(see ``docs/methodology.md``). The exposure map is an identifying assumption: it must be
chosen from domain knowledge, not learned from data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from graphdml.data import GraphData, drop_diagonal

__all__ = [
    "ExposureMap",
    "MatrixExposure",
    "MeanExposure",
    "SumExposure",
    "TwoHopExposure",
    "WeightedExposure",
    "resolve_exposures",
]


class ExposureMap:
    """Base class. Subclasses implement :meth:`matrix`."""

    name: str = "exposure"

    def matrix(self, data: GraphData) -> sp.csr_array:
        """The ``(n, n)`` operator ``E`` such that exposure ``Z = E @ T``."""
        raise NotImplementedError

    def units(self, binary_treatment: bool) -> str:
        """Plain-language unit of the peer-effect coefficient."""
        return "per unit of exposure"

    def __call__(self, data: GraphData, T: np.ndarray | None = None) -> np.ndarray:
        return self.matrix(data) @ (data.T if T is None else np.asarray(T, dtype=float))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SumExposure(ExposureMap):
    """``Z_i = sum of in-neighbors' treatments`` (``Z = A T``)."""

    name = "sum"

    def matrix(self, data: GraphData) -> sp.csr_array:
        return data.pattern

    def units(self, binary_treatment: bool) -> str:
        if binary_treatment:
            return "change in Y per additional treated neighbor"
        return "change in Y per unit increase in the sum of neighbors' treatments"


class MeanExposure(ExposureMap):
    """``Z_i = average of in-neighbors' treatments`` (0 for nodes without neighbors)."""

    name = "mean"

    def matrix(self, data: GraphData) -> sp.csr_array:
        return row_normalize(data.pattern)

    def units(self, binary_treatment: bool) -> str:
        if binary_treatment:
            return "change in Y going from 0% to 100% of neighbors treated"
        return "change in Y per unit increase in neighbors' average treatment"


class WeightedExposure(ExposureMap):
    """``Z_i = sum_j w_ij T_j`` using adjacency weights, optionally row-normalised."""

    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize

    @property
    def name(self) -> str:  # type: ignore[override]
        return "weighted_mean" if self.normalize else "weighted"

    def matrix(self, data: GraphData) -> sp.csr_array:
        W = data.adjacency
        return row_normalize(W) if self.normalize else W

    def units(self, binary_treatment: bool) -> str:
        return "change in Y per unit of weighted neighbor treatment"

    def __repr__(self) -> str:
        return f"WeightedExposure(normalize={self.normalize})"


class TwoHopExposure(ExposureMap):
    """Treatments of nodes at distance exactly two (summed or averaged).

    Used by :func:`graphdml.two_hop_test` to check the 1-hop interference assumption.
    """

    def __init__(self, normalize: str = "sum") -> None:
        if normalize not in ("sum", "mean"):
            raise ValueError("normalize must be 'sum' or 'mean'.")
        self.normalize = normalize

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"two_hop_{self.normalize}"

    def matrix(self, data: GraphData) -> sp.csr_array:
        P2 = two_hop_pattern(data.pattern)
        return row_normalize(P2) if self.normalize == "mean" else P2

    def units(self, binary_treatment: bool) -> str:
        return "change in Y per unit of two-hop exposure"

    def __repr__(self) -> str:
        return f"TwoHopExposure(normalize={self.normalize!r})"


class MatrixExposure(ExposureMap):
    """A user-supplied exposure operator ``E`` (sparse or dense ``(n, n)``)."""

    def __init__(self, matrix: Any, name: str = "custom", units: str = "per unit of exposure"):
        self._matrix = sp.csr_array(matrix, dtype=float)
        self._name = name
        self._units = units

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def matrix(self, data: GraphData) -> sp.csr_array:
        if self._matrix.shape != (data.n_nodes, data.n_nodes):
            raise ValueError("Exposure matrix shape does not match the number of nodes.")
        if self._matrix.diagonal().any():
            raise ValueError("Exposure matrix must have a zero diagonal (own treatment is T).")
        return self._matrix

    def units(self, binary_treatment: bool) -> str:
        return self._units

    def __repr__(self) -> str:
        return f"MatrixExposure(name={self._name!r})"


_REGISTRY = {
    "sum": SumExposure,
    "mean": MeanExposure,
    "weighted": WeightedExposure,
    "weighted_mean": lambda: WeightedExposure(normalize=True),
    "two_hop_sum": lambda: TwoHopExposure("sum"),
    "two_hop_mean": lambda: TwoHopExposure("mean"),
}


def resolve_exposures(spec: Any) -> list[ExposureMap]:
    """Turn ``None``, a name, an :class:`ExposureMap` or a list of those into a list."""
    if spec is None:
        return []
    items = spec if isinstance(spec, Sequence) and not isinstance(spec, str) else [spec]
    out: list[ExposureMap] = []
    for item in items:
        if isinstance(item, ExposureMap):
            out.append(item)
        elif isinstance(item, str) and item in _REGISTRY:
            out.append(_REGISTRY[item]())
        else:
            raise ValueError(
                f"Unknown exposure {item!r}. Use one of {sorted(_REGISTRY)} or an ExposureMap."
            )
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Exposure names must be unique, got {names}.")
    return out


def row_normalize(M: sp.csr_array) -> sp.csr_array:
    """Divide each row by its sum; all-zero rows stay zero."""
    s = np.asarray(M.sum(axis=1)).ravel()
    inv = np.divide(1.0, s, out=np.zeros_like(s, dtype=float), where=s != 0)
    return sp.csr_array(sp.diags_array(inv) @ M)


def two_hop_pattern(P: sp.csr_array) -> sp.csr_array:
    """Binary matrix of nodes at shortest-path distance exactly two (via in-edges)."""
    n = P.shape[0]
    P2 = sp.csr_array(P @ P)
    P2.data = np.ones_like(P2.data)
    P2 = drop_diagonal(P2 - P2.multiply(P))  # drop direct neighbors and self
    P2.eliminate_zeros()
    P2.data = np.ones_like(P2.data)
    P2.sort_indices()
    assert P2.shape == (n, n)
    return P2
