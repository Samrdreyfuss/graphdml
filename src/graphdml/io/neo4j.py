"""Neo4j integration: load graphs, write results back, and use GDS embeddings as features.

Requires the ``neo4j`` driver (``pip install graphdml[neo4j]``). GDS embeddings need the
Graph Data Science plugin on the server.

Safety: labels, relationship types and property names cannot be passed to Cypher as
parameters, so they are validated against ``[A-Za-z_][A-Za-z0-9_]*`` and backtick-quoted;
every value is passed as a parameter. Writes are batched with ``UNWIND`` and never modify
existing node properties: results live on a ``(:GDMLRun)`` node and its ``ESTIMATED_ON``
relationships, which :meth:`Neo4jResultWriter.delete_run` removes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from graphdml.data import GraphData
from graphdml.features import NeighborhoodFeatures

if TYPE_CHECKING:
    from graphdml.datasets import GraphDataset
    from graphdml.estimator import GraphDML

__all__ = [
    "GDSEmbeddings",
    "Neo4jGraphSource",
    "Neo4jResultWriter",
    "build_load_queries",
    "connect",
    "write_dataset",
]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INFLUENCE = ("undirected", "source_to_target", "target_to_source")
_RUN_LABEL = "GDMLRun"
_RUN_REL = "ESTIMATED_ON"


def _q(name: str) -> str:
    """Validate an identifier (label, type, property) and backtick-quote it."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(
            f"Invalid Neo4j identifier {name!r}: use letters, digits and underscores only."
        )
    return f"`{name}`"


def _neo4j():
    try:
        import neo4j
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Neo4j support needs the driver: pip install 'graphdml[neo4j]'"
        ) from e
    return neo4j


def connect(uri: str | None = None, auth: tuple[str, str] | None = None) -> Any:
    """Open a Neo4j driver. Defaults: ``$NEO4J_URI``, ``$NEO4J_USER``/``$NEO4J_PASSWORD``."""
    neo4j = _neo4j()
    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    if auth is None:
        auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    driver = neo4j.GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    return driver


class _Client:
    def __init__(self, driver: Any = None, *, uri: str | None = None,
                 auth: tuple[str, str] | None = None, database: str | None = None) -> None:
        self._owns_driver = driver is None
        self.driver = driver if driver is not None else connect(uri, auth)
        self.database = database

    def _run(self, query: str, write: bool = False, **params: Any) -> list[dict]:
        # Server notifications (e.g. "label does not exist" before the first run) are
        # expected for graphdml's own queries, so they are switched off here.
        with self.driver.session(database=self.database,
                                 notifications_min_severity="OFF") as session:
            work = session.execute_write if write else session.execute_read
            return work(lambda tx: [r.data() for r in tx.run(query, params)])

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------- loading
def build_load_queries(
    node_label: str,
    rel_type: str,
    *,
    features: Sequence[str],
    treatment: str,
    outcome: str,
    id_property: str | None = None,
) -> tuple[str, str]:
    """Cypher for :meth:`Neo4jGraphSource.load`: a node query and an edge query.

    The node query returns ``id`` plus one column per property; the edge query returns
    ``source`` and ``target`` (the relationship's start and end node ids).
    """
    label, rel = _q(node_label), _q(rel_type)
    key = f"n.{_q(id_property)}" if id_property else "elementId(n)"
    props = [*features, treatment, outcome]
    cols = ", ".join(f"n.{_q(p)} AS {_q(p)}" for p in dict.fromkeys(props))
    node_q = f"MATCH (n:{label}) RETURN {key} AS id, {cols}"
    a = f"a.{_q(id_property)}" if id_property else "elementId(a)"
    b = f"b.{_q(id_property)}" if id_property else "elementId(b)"
    edge_q = f"MATCH (a:{label})-[:{rel}]->(b:{label}) RETURN {a} AS source, {b} AS target"
    return node_q, edge_q


class Neo4jGraphSource(_Client):
    """Read a graph from Neo4j into :class:`~graphdml.GraphData`.

    Parameters
    ----------
    driver : neo4j.Driver, optional
        An existing driver; otherwise one is opened from ``uri``/``auth`` (or the
        ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD`` environment variables).
    database : str, optional
    """

    def load(
        self,
        node_label: str,
        rel_type: str,
        *,
        features: Sequence[str],
        treatment: str,
        outcome: str,
        id_property: str | None = None,
        influence: str = "undirected",
    ) -> GraphData:
        """Load all ``node_label`` nodes and the ``rel_type`` relationships among them.

        ``influence`` says how a relationship ``(a)-[:REL]->(b)`` transmits effects:
        ``"undirected"`` (both ways, e.g. FRIENDS_WITH), ``"source_to_target"`` (``a``
        influences ``b``) or ``"target_to_source"`` (``b`` influences ``a``, e.g. ``a``
        FOLLOWS ``b``). Node ids are ``id_property`` values if given (recommended; they
        should be unique), otherwise Neo4j element ids.
        """
        node_q, edge_q = build_load_queries(
            node_label, rel_type, features=features, treatment=treatment, outcome=outcome,
            id_property=id_property,
        )
        return self.load_query(node_q, edge_q, features=features, treatment=treatment,
                               outcome=outcome, influence=influence)

    def load_query(
        self,
        node_query: str,
        edge_query: str,
        *,
        features: Sequence[str],
        treatment: str,
        outcome: str,
        influence: str = "undirected",
        parameters: dict[str, Any] | None = None,
    ) -> GraphData:
        """Load from your own Cypher (e.g. one village, or a computed property).

        ``node_query`` must return a column ``id`` plus the named property columns;
        ``edge_query`` must return ``source`` and ``target`` ids. Edges whose endpoints
        are not returned by the node query are dropped.
        """
        if influence not in _INFLUENCE:
            raise ValueError(f"influence must be one of {_INFLUENCE}.")
        params = parameters or {}
        nodes = pd.DataFrame(self._run(node_query, **params))
        if nodes.empty:
            raise ValueError("The node query returned no rows.")
        needed = ["id", *features, treatment, outcome]
        missing_cols = [c for c in needed if c not in nodes.columns]
        if missing_cols:
            raise KeyError(f"Node query did not return columns {missing_cols}.")
        nulls = nodes[needed].isna().sum()
        if nulls.any():
            bad = nulls[nulls > 0].to_dict()
            raise ValueError(
                f"Null values in node properties {bad}. Filter or impute them in Cypher "
                "(e.g. coalesce(n.x, 0)) or in Neo4j before loading."
            )
        if nodes["id"].duplicated().any():
            raise ValueError("Node ids are not unique; choose a unique id_property.")

        edges = pd.DataFrame(self._run(edge_query, **params), columns=["source", "target"])
        known = edges["source"].isin(nodes["id"]) & edges["target"].isin(nodes["id"])
        edges = edges[known]
        if influence == "target_to_source":
            edges = edges.rename(columns={"source": "target", "target": "source"})
        return GraphData.from_pandas(
            nodes, edges, features=list(features), treatment=treatment, outcome=outcome,
            node_id="id", directed=influence != "undirected",
        )


# ---------------------------------------------------------------------------- writing
class Neo4jResultWriter(_Client):
    """Write a fitted :class:`~graphdml.GraphDML` back to Neo4j.

    Each call creates one ``(:GDMLRun {name, ...estimates})`` node and an
    ``(:GDMLRun)-[:ESTIMATED_ON {fold, t_hat, y_hat, res_t, ..., res_y}]->(n)``
    relationship to every focal node, so runs can be compared, visualised (focal nodes are
    exactly the ones with an ``ESTIMATED_ON`` edge) and deleted without touching your data.
    """

    def write(
        self,
        model: GraphDML,
        *,
        run_name: str,
        node_label: str,
        id_property: str | None = None,
        description: str = "",
        dry_run: bool = False,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Write the run. ``id_property`` must match what the data were loaded with.

        Returns a summary dict. With ``dry_run=True`` nothing is written and the summary
        includes the first rows that would be.
        """
        model._check_fitted()
        label = _q(node_label)
        if self.list_runs(run_name).shape[0]:
            raise ValueError(f"A run named {run_name!r} exists; delete_run() it or rename.")
        if id_property:
            dupes = self._run(
                f"MATCH (n:{label}) WITH n.{_q(id_property)} AS k, count(*) AS c "
                "WHERE c > 1 RETURN count(*) AS dupes"
            )[0]["dupes"]
            if dupes:
                raise ValueError(f"{dupes} {id_property} values are shared by several nodes.")

        frame = model.summary_frame()
        run_props: dict[str, Any] = {
            "name": run_name,
            "description": description,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "graphdml_version": _version(),
            "n_nodes": int(model.n_nodes_),
            "n_focal": int(model.n_focal_),
            "settings": json.dumps(_jsonable(model.get_params(deep=False))),
            "warnings": list(model.warnings_),
        }
        for name, row in frame.iterrows():
            key = _prop_key(name)
            for col in ("coef", "std_err", "ci_lower", "ci_upper", "p_value"):
                run_props[f"{key}_{col}"] = float(row[col])

        res = model.residuals_.rename(columns=_prop_key)
        cols = [c for c in res.columns if c not in ("node_id", "node_index")]
        rows = [
            {"id": _py(r["node_id"]), "props": {c: _py(r[c]) for c in cols}}
            for r in res.to_dict("records")
        ]
        summary = {"run": run_name, "focal_nodes": len(rows), "estimates": frame}
        if dry_run:
            summary["sample_rows"] = rows[:5]
            summary["run_properties"] = run_props
            return summary

        self._run(f"CREATE (r:{_RUN_LABEL}) SET r = $props", write=True, props=run_props)
        match = f"n.{_q(id_property)} = row.id" if id_property else "elementId(n) = row.id"
        query = (
            f"MATCH (r:{_RUN_LABEL} {{name: $name}}) "
            f"UNWIND $rows AS row MATCH (n:{label}) WHERE {match} "
            f"CREATE (r)-[e:{_RUN_REL}]->(n) SET e = row.props RETURN count(e) AS written"
        )
        written = 0
        for batch in _batches(rows, batch_size):
            written += self._run(query, write=True, name=run_name, rows=batch)[0]["written"]
        if written != len(rows):
            raise RuntimeError(
                f"Matched {written} of {len(rows)} focal nodes; check node_label and "
                "id_property match how the data were loaded."
            )
        summary["written"] = written
        return summary

    def list_runs(self, name: str | None = None) -> pd.DataFrame:
        """All runs (or one by name), newest first."""
        where = "WHERE r.name = $name " if name else ""
        rows = self._run(
            f"MATCH (r:{_RUN_LABEL}) {where}RETURN r {{.*}} AS run ORDER BY r.created_at DESC",
            name=name,
        )
        return pd.DataFrame([r["run"] for r in rows])

    def delete_run(self, run_name: str) -> int:
        """Delete a run node and its relationships. Returns the number of runs deleted."""
        rows = self._run(
            f"MATCH (r:{_RUN_LABEL} {{name: $name}}) DETACH DELETE r RETURN count(*) AS n",
            write=True, name=run_name,
        )
        return int(rows[0]["n"]) if rows else 0


def write_dataset(
    dataset: GraphDataset | GraphData,
    driver: Any,
    *,
    node_label: str,
    rel_type: str,
    id_property: str = "node_id",
    database: str | None = None,
    replace: bool = False,
    batch_size: int = 2000,
) -> dict[str, int]:
    """Load a dataset into Neo4j (for demos and tests).

    Nodes get ``id_property``, the covariates, and the treatment/outcome under their story
    names (e.g. ``flu_shot``, ``sick_days``). Undirected graphs get one relationship per
    pair; load them back with ``influence="undirected"``. ``replace=True`` first deletes
    every ``node_label`` node.
    """
    data = dataset.data if hasattr(dataset, "data") else dataset
    t_name = getattr(dataset, "treatment_name", "T")
    y_name = getattr(dataset, "outcome_name", "Y")
    client = _Client(driver, database=database)
    label, rel, key = _q(node_label), _q(rel_type), _q(id_property)
    for name in [*data.feature_names, t_name, y_name]:
        _q(name)

    if replace:
        while client._run(f"MATCH (n:{label}) WITH n LIMIT 10000 DETACH DELETE n "
                          "RETURN count(*) AS c", write=True)[0]["c"]:
            pass
    elif client._run(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]:
        raise ValueError(f"{node_label} nodes already exist; pass replace=True to overwrite.")
    client._run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE",
                write=True)

    nodes, edges = data.to_pandas()
    nodes = nodes.rename(columns={"T": t_name, "Y": y_name, "node_id": id_property})
    node_rows = [{k: _py(v) for k, v in r.items()} for r in nodes.to_dict("records")]
    for batch in _batches(node_rows, batch_size):
        client._run(f"UNWIND $rows AS row CREATE (n:{label}) SET n = row", write=True,
                    rows=batch)
    edge_rows = [{"s": _py(s), "t": _py(t)} for s, t in zip(edges["source"], edges["target"],
                                                             strict=True)]
    for batch in _batches(edge_rows, batch_size):
        client._run(
            f"UNWIND $rows AS row MATCH (a:{label} {{{key}: row.s}}), (b:{label} {{{key}: row.t}}) "
            f"CREATE (a)-[:{rel}]->(b)", write=True, rows=batch,
        )
    return {"nodes": len(node_rows), "relationships": len(edge_rows)}


# ---------------------------------------------------------------------------- GDS features
class GDSEmbeddings(BaseEstimator):
    """Featurizer: a base featurizer's output plus Neo4j GDS FastRP node embeddings.

    FastRP embeddings summarise each node's position in the graph (structure only, never
    treatment or outcome), so they can stand in for unobserved structural confounders such
    as centrality or community. Rows are aligned to ``data.node_ids``, which must be the
    ``id_property`` values used when loading.

    Parameters
    ----------
    source : Neo4jGraphSource
    node_label, rel_type, id_property : the same graph that was loaded
    dimension : embedding size
    base : featurizer whose output is kept alongside the embeddings
        (default :class:`~graphdml.NeighborhoodFeatures`; ``None`` for embeddings only)
    random_seed : FastRP seed (embeddings are deterministic given the seed)
    """

    def __init__(self, source: Neo4jGraphSource, node_label: str, rel_type: str,
                 id_property: str, dimension: int = 16, base: Any = "default",
                 random_seed: int = 42) -> None:
        self.source = source
        self.node_label = node_label
        self.rel_type = rel_type
        self.id_property = id_property
        self.dimension = dimension
        self.base = base
        self.random_seed = random_seed

    def embeddings(self, data: GraphData) -> np.ndarray:
        label, rel, key = _q(self.node_label), _q(self.rel_type), _q(self.id_property)
        graph = f"graphdml_{os.getpid()}_{id(self)}"
        try:
            self.source._run(
                f"MATCH (a:{label}) OPTIONAL MATCH (a)-[:{rel}]->(b:{label}) "
                "WITH gds.graph.project($g, a, b, {}, {undirectedRelationshipTypes: ['*']}) AS p "
                "RETURN p.nodeCount AS nodes",
                write=True, g=graph,
            )
            rows = self.source._run(
                "CALL gds.fastRP.stream($g, {embeddingDimension: $dim, randomSeed: $seed}) "
                f"YIELD nodeId, embedding RETURN gds.util.asNode(nodeId).{key} AS id, embedding",
                g=graph, dim=int(self.dimension), seed=int(self.random_seed),
            )
        except Exception as e:
            if "gds" in str(e).lower() and "unknown function" in str(e).lower():
                raise RuntimeError("GDSEmbeddings needs the Graph Data Science plugin.") from e
            raise
        finally:
            try:
                self.source._run("CALL gds.graph.drop($g, false) YIELD graphName "
                                 "RETURN graphName", write=True, g=graph)
            except Exception:
                pass
        emb = pd.DataFrame({"id": [r["id"] for r in rows],
                            "e": [r["embedding"] for r in rows]}).set_index("id")["e"]
        missing = pd.Index(data.node_ids).difference(emb.index)
        if len(missing):
            raise ValueError(f"{len(missing)} nodes have no embedding (ids not found in Neo4j).")
        return np.vstack(emb.loc[data.node_ids].to_numpy())

    def transform(self, data: GraphData) -> np.ndarray:
        base = NeighborhoodFeatures() if self.base == "default" else self.base
        emb = self.embeddings(data)
        return emb if base is None else np.hstack([base.transform(data), emb])


# ---------------------------------------------------------------------------- helpers
def _batches(rows: list, size: int) -> Iterable[list]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _prop_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        out[k] = v if isinstance(v, str | int | float | bool | type(None)) else repr(v)
    return out


def _version() -> str:
    from graphdml import __version__

    return __version__
