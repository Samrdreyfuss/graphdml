"""Featurizers: turn ``(X, graph)`` into a tabular matrix any scikit-learn model can use.

The nuisance functions ``E[T | X, A]`` and ``E[Y | X, A]`` depend on the graph only
through covariates (never neighbors' T or Y), so featurizing the full graph once is
leakage-free. Note that ``E[Y | X, A]`` generally depends on *two-hop* covariates even
under one-hop interference, because it contains ``alpha * E @ m(X, A)`` and ``m`` itself
depends on one-hop covariates. That is why :class:`NeighborhoodFeatures` defaults to
``hops=2``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator

from graphdml.data import GraphData
from graphdml.exposure import row_normalize

__all__ = [
    "NeighborhoodFeatures",
    "OwnFeatures",
    "PrecomputedFeatures",
    "resolve_featurizer",
]

_AGGS = ("mean", "sum", "max", "min")


class NeighborhoodFeatures(BaseEstimator):
    """Own covariates plus aggregated neighbor covariates over one or more hops.

    Hop 1 applies each aggregation in ``aggs`` to in-neighbors' covariates. Each further
    hop averages the previous hop's features over in-neighbors (mean propagation, as in
    SGC/SIGN-style graph features), so ``aggs=("mean",), hops=2`` yields
    ``[X, Ā X, Ā² X]`` with ``Ā`` the row-normalised adjacency.

    Parameters
    ----------
    aggs : tuple of {"mean", "sum", "max", "min"}, default=("mean", "max", "min")
    hops : int, default=2
    include_degree : bool, default=True
        Append each node's in-degree as a feature.
    """

    def __init__(
        self,
        aggs: tuple[str, ...] = ("mean", "max", "min"),
        hops: int = 2,
        include_degree: bool = True,
    ) -> None:
        self.aggs = aggs
        self.hops = hops
        self.include_degree = include_degree

    def transform(self, data: GraphData) -> np.ndarray:
        blocks, _ = self._build(data)
        return np.hstack(blocks)

    def get_feature_names_out(self, data: GraphData) -> np.ndarray:
        _, names = self._build(data, names_only=True)
        return np.asarray(names, dtype=object)

    def _build(self, data: GraphData, names_only: bool = False):
        bad = [a for a in self.aggs if a not in _AGGS]
        if bad or not self.aggs:
            raise ValueError(f"aggs must be a non-empty subset of {_AGGS}, got {self.aggs}.")
        if self.hops < 0:
            raise ValueError("hops must be >= 0.")
        names = list(data.feature_names)
        blocks: list[np.ndarray] = [] if names_only else [data.X]
        if self.hops >= 1:
            P = data.pattern
            P_mean = row_normalize(P)
            level_names = [f"{agg}_nbr({c})" for agg in self.aggs for c in data.feature_names]
            names += level_names
            if not names_only:
                level = np.hstack([_aggregate(P, P_mean, data.X, agg) for agg in self.aggs])
                blocks.append(level)
            for _ in range(2, self.hops + 1):
                level_names = [f"mean_nbr({c})" for c in level_names]
                names += level_names
                if not names_only:
                    level = P_mean @ level
                    blocks.append(level)
        if self.include_degree:
            names.append("degree")
            if not names_only:
                blocks.append(data.degree.astype(float)[:, None])
        return blocks, names


class OwnFeatures(BaseEstimator):
    """Only the node's own covariates (ignores the network; i.i.d. DML baseline)."""

    def transform(self, data: GraphData) -> np.ndarray:
        return data.X

    def get_feature_names_out(self, data: GraphData) -> np.ndarray:
        return np.asarray(data.feature_names, dtype=object)


class PrecomputedFeatures(BaseEstimator):
    """A fixed ``(n_nodes, n_features)`` matrix, e.g. graph embeddings computed elsewhere.

    The features must be functions of covariates and graph structure only, never of
    treatments or outcomes.
    """

    def __init__(self, features: Any, names: Any = None) -> None:
        self.features = features
        self.names = names

    def transform(self, data: GraphData) -> np.ndarray:
        F = np.asarray(self.features, dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        if F.shape[0] != data.n_nodes:
            raise ValueError(f"Expected {data.n_nodes} rows of features, got {F.shape[0]}.")
        return F

    def get_feature_names_out(self, data: GraphData) -> np.ndarray:
        k = self.transform(data).shape[1]
        names = self.names if self.names is not None else [f"f{j}" for j in range(k)]
        return np.asarray(names, dtype=object)


def resolve_featurizer(spec: Any) -> Any:
    if spec is None:
        return NeighborhoodFeatures()
    if isinstance(spec, str):
        if spec == "own":
            return OwnFeatures()
        if spec == "neighborhood":
            return NeighborhoodFeatures()
        raise ValueError(f"Unknown featurizer {spec!r}; use 'own', 'neighborhood' or an object.")
    if not hasattr(spec, "transform"):
        raise TypeError("featurizer must have a transform(data) method.")
    return spec


def _aggregate(P: sp.csr_array, P_mean: sp.csr_array, X: np.ndarray, agg: str) -> np.ndarray:
    if agg == "mean":
        return P_mean @ X
    if agg == "sum":
        return P @ X
    ufunc = np.maximum if agg == "max" else np.minimum
    out = np.zeros_like(X)
    deg = np.diff(P.indptr)
    has = deg > 0
    if has.any():
        vals = X[P.indices]
        out[has] = ufunc.reduceat(vals, P.indptr[:-1][has], axis=0)
    return out
