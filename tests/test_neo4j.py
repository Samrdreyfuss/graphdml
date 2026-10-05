"""Neo4j integration.

Unit tests run everywhere. Integration tests need a live server:

    GRAPHDML_NEO4J_URI=bolt://localhost:7687 GRAPHDML_NEO4J_PASSWORD=... pytest tests/test_neo4j.py
"""

import os
import uuid

import numpy as np
import pytest

from graphdml import GraphData, GraphDML

neo4j_io = pytest.importorskip("graphdml.io.neo4j")
pytest.importorskip("neo4j")


# ---------------------------------------------------------------- unit (no server)
@pytest.mark.parametrize(
    "bad", ["Person`) DETACH DELETE n //", "has space", "1starts_with_digit", "", "a-b"]
)
def test_identifiers_reject_injection(bad):
    with pytest.raises(ValueError, match="Invalid Neo4j identifier"):
        neo4j_io._q(bad)


def test_build_load_queries_quotes_everything():
    node_q, edge_q = neo4j_io.build_load_queries(
        "Person", "KNOWS", features=["age"], treatment="t", outcome="y", id_property="pid"
    )
    assert node_q == ("MATCH (n:`Person`) RETURN n.`pid` AS id, n.`age` AS `age`, "
                      "n.`t` AS `t`, n.`y` AS `y`")
    assert edge_q == ("MATCH (a:`Person`)-[:`KNOWS`]->(b:`Person`) "
                      "RETURN a.`pid` AS source, b.`pid` AS target")
    with pytest.raises(ValueError):
        neo4j_io.build_load_queries("Person", "KNOWS", features=["x; DROP"], treatment="t",
                                    outcome="y")


# ---------------------------------------------------------------- integration
@pytest.fixture(scope="module")
def driver():
    uri = os.environ.get("GRAPHDML_NEO4J_URI")
    if not uri:
        pytest.skip("set GRAPHDML_NEO4J_URI (and GRAPHDML_NEO4J_PASSWORD) to run")
    drv = neo4j_io.connect(uri, (os.environ.get("GRAPHDML_NEO4J_USER", "neo4j"),
                                 os.environ.get("GRAPHDML_NEO4J_PASSWORD", "")))
    yield drv
    drv.close()


@pytest.fixture
def label(driver):
    name = f"GdmlTest{uuid.uuid4().hex[:8]}"
    yield name
    client = neo4j_io._Client(driver)
    client._run(f"MATCH (n:`{name}`) DETACH DELETE n", write=True)
    client._run("MATCH (r:GDMLRun) WHERE r.name STARTS WITH $p DETACH DELETE r", write=True,
                p=name)


def _sorted(data):
    o = np.argsort(data.node_ids)
    return data.X[o], data.T[o], data.Y[o], data.adjacency[o][:, o]


def test_roundtrip_undirected(driver, label):
    from graphdml.datasets import make_flu_town

    ds = make_flu_town(800, random_state=1)
    neo4j_io.write_dataset(ds, driver, node_label=label, rel_type="NEAR")
    data = neo4j_io.Neo4jGraphSource(driver).load(
        label, "NEAR", features=list(ds.data.feature_names), treatment="flu_shot",
        outcome="sick_days", id_property="node_id",
    )
    X, T, Y, A = _sorted(data)
    np.testing.assert_allclose(X, ds.data.X)
    np.testing.assert_array_equal(T, ds.data.T)
    np.testing.assert_allclose(Y, ds.data.Y)
    assert (A != ds.data.adjacency).nnz == 0


@pytest.mark.parametrize("influence", ["source_to_target", "target_to_source"])
def test_roundtrip_directed(driver, label, influence):
    rng = np.random.RandomState(0)
    edges = np.array([(0, 1), (1, 2), (3, 2), (4, 0)])
    original = GraphData.from_edges(rng.standard_normal((5, 2)), [0, 1, 0, 1, 1],
                                    rng.standard_normal(5), edges, directed=True)
    neo4j_io.write_dataset(original, driver, node_label=label, rel_type="FOLLOWS")
    data = neo4j_io.Neo4jGraphSource(driver).load(
        label, "FOLLOWS", features=["X0", "X1"], treatment="T", outcome="Y",
        id_property="node_id", influence=influence,
    )
    A = _sorted(data)[3]
    expected = original.adjacency if influence == "source_to_target" else original.adjacency.T
    assert (A != expected).nnz == 0


def test_null_properties_are_reported(driver, label):
    client = neo4j_io._Client(driver)
    client._run(f"CREATE (:`{label}` {{node_id: 1, x: 1.0, t: 1, y: 2.0}}), "
                f"(:`{label}` {{node_id: 2, t: 0, y: 1.0}})", write=True)
    with pytest.raises(ValueError, match="Null values"):
        neo4j_io.Neo4jGraphSource(driver).load(label, "R", features=["x"], treatment="t",
                                               outcome="y", id_property="node_id")


def test_write_list_and_delete_runs(driver, label):
    from graphdml.datasets import make_flu_town

    ds = make_flu_town(1500, random_state=2)
    neo4j_io.write_dataset(ds, driver, node_label=label, rel_type="NEAR")
    data = neo4j_io.Neo4jGraphSource(driver).load(
        label, "NEAR", features=list(ds.data.feature_names), treatment="flu_shot",
        outcome="sick_days", id_property="node_id",
    )
    model = GraphDML(exposure="sum", random_state=0).fit(data)
    writer = neo4j_io.Neo4jResultWriter(driver)
    run = f"{label}-run"

    dry = writer.write(model, run_name=run, node_label=label, id_property="node_id",
                       dry_run=True)
    assert dry["focal_nodes"] == model.n_focal_ and writer.list_runs(run).empty

    out = writer.write(model, run_name=run, node_label=label, id_property="node_id")
    assert out["written"] == model.n_focal_
    runs = writer.list_runs(run)
    assert runs.loc[0, "direct_coef"] == pytest.approx(model.ade_)
    n_edges = writer._run(
        f"MATCH (:GDMLRun {{name: $r}})-[e:ESTIMATED_ON]->(n:`{label}`) RETURN count(e) AS c",
        r=run)[0]["c"]
    assert n_edges == model.n_focal_
    with pytest.raises(ValueError, match="exists"):
        writer.write(model, run_name=run, node_label=label, id_property="node_id")
    assert writer.delete_run(run) == 1 and writer.list_runs(run).empty


def test_gds_embeddings(driver, label):
    client = neo4j_io._Client(driver)
    try:
        client._run("RETURN gds.version() AS v")
    except Exception:
        pytest.skip("GDS plugin not installed")
    from graphdml.datasets import make_flu_town

    ds = make_flu_town(600, random_state=3)
    neo4j_io.write_dataset(ds, driver, node_label=label, rel_type="NEAR")
    src = neo4j_io.Neo4jGraphSource(driver)
    data = src.load(label, "NEAR", features=list(ds.data.feature_names), treatment="flu_shot",
                    outcome="sick_days", id_property="node_id")
    emb = neo4j_io.GDSEmbeddings(src, label, "NEAR", "node_id", dimension=8, base=None)
    E1, E2 = emb.transform(data), emb.transform(data)
    assert E1.shape == (600, 8)
    np.testing.assert_allclose(E1, E2)  # deterministic given the seed
