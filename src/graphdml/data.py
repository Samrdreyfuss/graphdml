"""The :class:`GraphData` container: covariates, treatment, outcome and the network."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graphdml.exceptions import GraphDMLWarning

__all__ = ["GraphData"]


@dataclass(frozen=True, eq=False, repr=False)
class GraphData:
    """Node covariates, treatment and outcome, plus the network that connects the nodes.

    Parameters
    ----------
    X : array of shape (n_nodes, n_features)
        Pre-treatment covariates. Must be numeric and finite.
    T : array of shape (n_nodes,)
        Treatment, binary (0/1) or continuous.
    Y : array of shape (n_nodes,)
        Outcome.
    adjacency : sparse matrix of shape (n_nodes, n_nodes)
        ``adjacency[i, j] != 0`` means node ``j`` can influence node ``i`` (``j`` is an
        in-neighbor of ``i``). Undirected graphs must be symmetric. Values are only used as
        weights by :class:`~graphdml.WeightedExposure`; everything else uses the pattern.
    feature_names : sequence of str, optional
    node_ids : array of shape (n_nodes,), optional
        External identifiers such as Neo4j element ids. Defaults to ``0..n_nodes-1``.
    directed : bool, default=False

    Notes
    -----
    Most users build this with :meth:`from_edges`, :meth:`from_pandas` or
    :meth:`from_networkx`. Self-loops are dropped (a node does not "influence itself"
    through the network; its own treatment enters the model directly).
    """

    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    adjacency: Any
    feature_names: Sequence[str] | None = None
    node_ids: np.ndarray | None = None
    directed: bool = False

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
        n = X.shape[0]
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; impute or drop them first.")
        T = _as_vector(self.T, n, "T")
        Y = _as_vector(self.Y, n, "Y")

        A = sp.csr_array(self.adjacency, dtype=float, copy=True)
        if A.shape != (n, n):
            raise ValueError(f"adjacency must have shape ({n}, {n}), got {A.shape}.")
        A.sum_duplicates()
        if A.diagonal().any():
            warnings.warn("Dropping self-loops from adjacency.", GraphDMLWarning, stacklevel=3)
            A.setdiag(0)
        A.eliminate_zeros()
        A.sort_indices()
        if not self.directed:
            diff = abs(A - A.T)
            if diff.nnz and diff.max() > 1e-10:
                raise ValueError(
                    "adjacency is not symmetric. Pass directed=True for directed graphs "
                    "(adjacency[i, j] != 0 means j influences i)."
                )

        if self.feature_names is None:
            names = tuple(f"X{k}" for k in range(X.shape[1]))
        else:
            names = tuple(str(s) for s in self.feature_names)
            if len(names) != X.shape[1]:
                raise ValueError(f"Got {len(names)} feature_names for {X.shape[1]} features.")

        ids = np.arange(n) if self.node_ids is None else np.asarray(self.node_ids)
        if ids.shape != (n,):
            raise ValueError(f"node_ids must have shape ({n},), got {ids.shape}.")
        if len(pd.unique(ids)) != n:
            raise ValueError("node_ids must be unique.")

        for name, value in [
            ("X", X),
            ("T", T),
            ("Y", Y),
            ("adjacency", A),
            ("feature_names", names),
            ("node_ids", ids),
        ]:
            object.__setattr__(self, name, value)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_edges(
        cls,
        X: Any,
        T: Any,
        Y: Any,
        edges: Any,
        *,
        weights: Any = None,
        directed: bool = False,
        feature_names: Sequence[str] | None = None,
        node_ids: Any = None,
    ) -> GraphData:
        """Build from an ``(n_edges, 2)`` array of ``(source, target)`` node positions.

        ``source`` influences ``target``. For undirected graphs the order is irrelevant.
        Duplicate edges keep their first occurrence; self-loops are dropped.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        A = _adjacency_from_edges(n, edges, weights=weights, directed=directed)
        return cls(X, T, Y, A, feature_names=feature_names, node_ids=node_ids, directed=directed)

    @classmethod
    def from_pandas(
        cls,
        nodes: pd.DataFrame,
        edges: pd.DataFrame | None,
        *,
        features: Sequence[str],
        treatment: str,
        outcome: str,
        node_id: str | None = None,
        source: str = "source",
        target: str = "target",
        weight: str | None = None,
        directed: bool = False,
    ) -> GraphData:
        """Build from a node table and an edge table.

        ``node_id`` names the id column of ``nodes`` (defaults to the index). ``edges``
        refers to nodes by those ids; ``source`` influences ``target``.
        """
        ids = nodes.index.to_numpy() if node_id is None else nodes[node_id].to_numpy()
        missing = [c for c in [*features, treatment, outcome] if c not in nodes.columns]
        if missing:
            raise KeyError(f"Columns not found in nodes: {missing}")
        if edges is None or len(edges) == 0:
            edge_pos = np.empty((0, 2), dtype=int)
            w = None
        else:
            index = pd.Index(ids)
            src = index.get_indexer(edges[source])
            dst = index.get_indexer(edges[target])
            unknown = (src < 0) | (dst < 0)
            if unknown.any():
                bad = edges.loc[unknown, [source, target]].head(5).to_numpy().tolist()
                raise ValueError(f"{unknown.sum()} edges reference unknown node ids, e.g. {bad}")
            edge_pos = np.column_stack([src, dst])
            w = None if weight is None else edges[weight].to_numpy(dtype=float)
        return cls.from_edges(
            nodes[list(features)].to_numpy(dtype=float),
            nodes[treatment].to_numpy(),
            nodes[outcome].to_numpy(),
            edge_pos,
            weights=w,
            directed=directed,
            feature_names=list(features),
            node_ids=ids,
        )

    @classmethod
    def from_networkx(
        cls,
        graph: Any,
        *,
        features: Sequence[str],
        treatment: str,
        outcome: str,
        weight: str | None = None,
    ) -> GraphData:
        """Build from a networkx graph whose nodes carry the given attributes.

        For a ``DiGraph``, an edge ``u -> v`` means ``u`` influences ``v``.
        """
        nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        rows = [
            (u, v, d.get(weight, 1.0) if weight else 1.0) for u, v, d in graph.edges(data=True)
        ]
        edges = pd.DataFrame(rows, columns=["source", "target", "weight"])
        return cls.from_pandas(
            nodes,
            edges,
            features=features,
            treatment=treatment,
            outcome=outcome,
            weight="weight" if weight else None,
            directed=graph.is_directed(),
        )

    # ------------------------------------------------------------------ accessors
    @property
    def n_nodes(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_edges(self) -> int:
        """Number of edges (unordered pairs for undirected graphs)."""
        nnz = self.adjacency.nnz
        return nnz if self.directed else nnz // 2

    @cached_property
    def pattern(self) -> sp.csr_array:
        """Binary adjacency: ``pattern[i, j] == 1`` iff ``j`` is an in-neighbor of ``i``."""
        P = self.adjacency.copy()
        P.data = np.ones_like(P.data)
        return P

    @cached_property
    def degree(self) -> np.ndarray:
        """In-degree of every node (number of nodes that can influence it)."""
        return np.diff(self.adjacency.indptr)

    @property
    def treatment_is_binary(self) -> bool:
        return bool(np.isin(self.T, (0.0, 1.0)).all())

    def with_values(self, *, T: Any = None, Y: Any = None, X: Any = None) -> GraphData:
        """Return a copy with the treatment, outcome and/or covariates replaced."""
        changes: dict[str, Any] = {}
        if T is not None:
            changes["T"] = T
        if Y is not None:
            changes["Y"] = Y
        if X is not None:
            changes["X"] = X
        return replace(self, **changes)

    def to_pandas(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(nodes, edges)`` DataFrames. Edges have ``source`` influencing ``target``."""
        nodes = pd.DataFrame(self.X, columns=list(self.feature_names))
        nodes.insert(0, "node_id", self.node_ids)
        nodes["T"] = self.T
        nodes["Y"] = self.Y
        coo = self.adjacency.tocoo()
        targets, sources, weights = coo.row, coo.col, coo.data
        if not self.directed:
            keep = sources < targets
            targets, sources, weights = targets[keep], sources[keep], weights[keep]
        edges = pd.DataFrame(
            {
                "source": self.node_ids[sources],
                "target": self.node_ids[targets],
                "weight": weights,
            }
        )
        return nodes, edges

    def __repr__(self) -> str:
        kind = "binary" if self.treatment_is_binary else "continuous"
        return (
            f"GraphData(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"n_features={self.n_features}, treatment={kind}, directed={self.directed})"
        )


def _as_vector(values: Any, n: int, name: str) -> np.ndarray:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got {v.shape[0]}.")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return v


def _adjacency_from_edges(
    n: int, edges: Any, *, weights: Any = None, directed: bool = False
) -> sp.csr_array:
    edges = np.asarray(edges)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (n_edges, 2), got {edges.shape}.")
    edges = edges.astype(np.int64)
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError(f"edge endpoints must be node positions in [0, {n}).")
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(edges),):
        raise ValueError("weights must have one entry per edge.")

    src, dst = edges[:, 0], edges[:, 1]
    loops = src == dst
    if loops.any():
        warnings.warn(
            f"Dropping {loops.sum()} self-loop edges.", GraphDMLWarning, stacklevel=3
        )
        src, dst, w = src[~loops], dst[~loops], w[~loops]

    if not directed:
        # Canonicalise each pair before de-duplicating so both directions share one weight.
        src, dst = np.minimum(src, dst), np.maximum(src, dst)
    _, first = np.unique(src * n + dst, return_index=True)
    first.sort()
    src, dst, w = src[first], dst[first], w[first]

    # source influences target  =>  adjacency[target, source]
    rows, cols = dst, src
    if not directed:
        rows, cols, w = np.concatenate([dst, src]), np.concatenate([src, dst]), np.tile(w, 2)
    return sp.csr_array((w, (rows, cols)), shape=(n, n))
