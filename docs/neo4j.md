# Neo4j integration

```bash
pip install -e ".[neo4j]"
docker compose up -d                                  # Neo4j 5 + Graph Data Science
python -m graphdml.demo --password graphdml-demo      # seed, estimate, write back
```

If ports 7474/7687 are taken by another Neo4j, run with
`GRAPHDML_NEO4J_HTTP_PORT=7476 GRAPHDML_NEO4J_BOLT_PORT=7689 docker compose up -d`, then
pass `--uri bolt://localhost:7689`.

## Load a graph

```python
from graphdml import GraphDML
from graphdml.io.neo4j import Neo4jGraphSource

src = Neo4jGraphSource(uri="bolt://localhost:7687", auth=("neo4j", "..."))
data = src.load(
    "Customer", "FRIENDS_WITH",
    features=["age", "tenure", "engagement"], treatment="got_coupon", outcome="spend",
    id_property="customer_id",          # a unique property; recommended
    influence="undirected",             # or "source_to_target" / "target_to_source"
)
model = GraphDML(exposure="mean").fit(data)
```

`influence` states how a relationship `(a)-[:REL]->(b)` carries effects. Friendships run both
ways (`"undirected"`). `(a)-[:FOLLOWS]->(b)` usually means `b` influences `a`
(`"target_to_source"`).

For anything more specific (a subgraph, computed properties, filtering), write the Cypher
yourself:

```python
data = src.load_query(
    "MATCH (n:Customer {region: $region}) "
    "RETURN n.customer_id AS id, n.age AS age, coalesce(n.tenure, 0) AS tenure, "
    "n.got_coupon AS got_coupon, n.spend AS spend",
    "MATCH (a:Customer {region: $region})-[:FRIENDS_WITH]->(b:Customer {region: $region}) "
    "RETURN a.customer_id AS source, b.customer_id AS target",
    features=["age", "tenure"], treatment="got_coupon", outcome="spend",
    parameters={"region": "EU"},
)
```

Null properties raise an error that names the columns. Impute in Cypher with `coalesce`, or
in the database.

## Write results back

```python
from graphdml.io.neo4j import Neo4jResultWriter

writer = Neo4jResultWriter(driver=src.driver)
writer.write(model, run_name="coupon-q3", node_label="Customer", id_property="customer_id",
             dry_run=True)   # inspect first
writer.write(model, run_name="coupon-q3", node_label="Customer", id_property="customer_id")
writer.list_runs()            # estimates, CIs, settings, warnings per run
writer.delete_run("coupon-q3")
```

Results never overwrite your node properties. Each run is a node:

```
(:GDMLRun {name, created_at, direct_coef, direct_std_err, direct_ci_lower, ...,
           peer_mean_coef, ..., n_focal, settings, warnings})
   -[:ESTIMATED_ON {fold, t_hat, y_hat, res_t, res_peer_mean, res_y}]->(focal node)
```

The focal nodes, which make up the effective sample, are exactly the nodes with an
`ESTIMATED_ON` relationship. That makes them easy to style in Bloom or Browser:

```cypher
MATCH (r:GDMLRun {name: 'coupon-q3'})-[e:ESTIMATED_ON]->(n)-[:FRIENDS_WITH]-(m)
RETURN n, m LIMIT 300
```

## Graph Data Science embeddings as features

```python
from graphdml.io.neo4j import GDSEmbeddings

features = GDSEmbeddings(src, "Customer", "FRIENDS_WITH", "customer_id", dimension=16)
model = GraphDML(exposure="mean", featurizer=features).fit(data)
```

FastRP embeddings are computed in the database from graph structure only (never from
treatments or outcomes), so they are safe nuisance features. By default they are appended
to the standard neighborhood features. They can help when position in the network, such as
centrality or community, confounds treatment and outcome.

## Safety

* Labels, relationship types and property names are validated (`[A-Za-z_][A-Za-z0-9_]*`) and
  backtick-quoted. All values are passed as query parameters.
* Writes are batched with `UNWIND`, refuse to overwrite an existing run name, and check
  that `id_property` values are unique before writing.
* Element ids can be reused by Neo4j after a node is deleted. Prefer a stable
  `id_property` with a uniqueness constraint.
