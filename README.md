# graphdml

**Causal effects on networks with double machine learning.** graphdml estimates how much a
treatment changes a node's own outcome (the **direct effect**) and how much it changes its
neighbors' outcomes (the **peer effect**), with confidence intervals that account for
network dependence. It works with any scikit-learn model and reads from and writes back to
Neo4j.

It implements and extends *Graph Machine Learning based Doubly Robust Estimator for Network
Causal Effects* (Khatami, Parikh, Chen, Roy and Salimi,
[AISTATS 2025](https://proceedings.mlr.press/v258/khatami25a.html)).

> **Status: alpha.** The estimator, datasets, diagnostics, validation suite and Neo4j
> integration are in place. GNN learners come next (see the [roadmap](#roadmap)).

## Why networks need their own method

Standard causal machine learning assumes units are independent. On a network, three things
break that:

* **Spillovers.** My outcome depends on my friends' treatments, not only mine.
* **Network confounding.** My friends' characteristics affect both my treatment and my
  outcome, and adjusting for my own covariates does not remove that.
* **Dependence.** Neighbors' data are correlated, so the usual standard errors are wrong.

graphdml fits the partially linear network model

```
T = m(X, A) + e_T
Y = θ·T + α·(E T) + g(X, A) + e_Y
```

where `E T` is a known **exposure map**, such as the number or share of treated neighbors.
The functions `m` and `g` are learned from own *and* neighbor covariates with any
scikit-learn model. θ is the average direct effect and α the average peer effect.

## Install

From a clone of this repository:

```bash
pip install -e .            # core: numpy, scipy, pandas, scikit-learn
pip install -e ".[dev]"     # + tests, benchmarks, validation tools
```

## Quickstart

```python
from graphdml import GraphDML
from graphdml.datasets import make_flu_town

ds = make_flu_town()     # 5,000 residents; flu shots; true effects -2.0 and -0.5 sick days
model = GraphDML(exposure="sum", random_state=0).fit(ds.data)
print(model.summary())
```

```
GraphDML: direct and peer effects
══════════════════════════════════════════════════════════════════════════════
Treatment: binary   Exposure: sum   Score: iv-type
Folds: 5 × 1 rep   Aggregation: dml2   Nuisance training: buffered
Nodes: 5,000   Focal nodes (effective sample size): 782 (15.6%)

                    coef   std err        z    P>|z|      [2.5%    97.5%]
direct            -1.879    0.1275   -14.73    0.000     -2.129    -1.629
peer:sum         -0.4341    0.0689    -6.30    0.000    -0.5692   -0.2991

  direct:       change in Y when own treatment goes 0 → 1
  peer:sum:     change in Y per additional treated neighbor

Diagnostics
  Outcome model R² (out-of-fold, focal nodes) ....... 0.726
  Treatment model AUC (out-of-fold, focal nodes) .... 0.805
  ...
Identifying assumptions (cannot be verified from data alone)
  • Exposure map: Y depends on others' treatments only through 'sum'.
  ...
```

Compare it with methods that ignore the network:

```python
from graphdml.baselines import compare_methods
compare_methods(ds.data, exposure="sum", truth=ds.truth)
```

| method | effect | estimate | 95% CI | truth |
|---|---|---|---|---|
| Naive OLS (no network) | direct | −2.39 | [−2.52, −2.26] | −2.0 ✗ |
| OLS + exposure (no neighbor covariates) | peer | −0.66 | [−0.68, −0.63] | −0.5 ✗ |
| DML, own covariates only | direct | −2.41 | [−2.54, −2.28] | −2.0 ✗ |
| **GraphDML** | direct | **−1.88** | [−2.13, −1.63] | −2.0 ✓ |
| **GraphDML** | peer | **−0.43** | [−0.57, −0.30] | −0.5 ✓ |

The naive methods credit the flu shot with the healthiness of the neighborhood, because
health-conscious people cluster together.

## Your own data

```python
from graphdml import GraphData, GraphDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

data = GraphData.from_pandas(
    nodes_df, edges_df,                       # edges: source influences target
    features=["age", "income"], treatment="treated", outcome="spend", node_id="user_id",
)
model = GraphDML(
    model_y=RandomForestRegressor(min_samples_leaf=20),   # any scikit-learn model
    model_t=RandomForestClassifier(min_samples_leaf=20),
    exposure="mean",                                       # share of treated neighbors
).fit(data)
```

`GraphData.from_edges` and `GraphData.from_networkx` also work.

### From Neo4j

```python
from graphdml.io.neo4j import Neo4jGraphSource, Neo4jResultWriter

src = Neo4jGraphSource(uri="bolt://localhost:7687", auth=("neo4j", "..."))
data = src.load("Customer", "FRIENDS_WITH", features=["age", "tenure"],
                treatment="got_coupon", outcome="spend", id_property="customer_id")
model = GraphDML(exposure="mean").fit(data)
Neo4jResultWriter(driver=src.driver).write(model, run_name="coupon-q3",
                                           node_label="Customer", id_property="customer_id")
```

Each run becomes a `(:GDMLRun)` node linked to the focal nodes it was estimated on, with
per-node predictions and residuals, ready to explore in Browser or Bloom. To try it locally,
run `docker compose up -d && python -m graphdml.demo --password graphdml-demo`. See
[docs/neo4j.md](docs/neo4j.md) for directed graphs, custom Cypher and GDS embeddings. Results are available as a
DataFrame via `model.summary_frame()`, and per-node residuals via `model.residuals_`.

## Learn with the example datasets

| Dataset | Story | Teaches |
|---|---|---|
| `make_flu_town` | flu shots and sick days among neighbors | direct vs peer effects; network confounding |
| `make_referral_app` | promo coupons in a social app with influencers | choosing the exposure map (`"sum"` vs `"mean"`) |
| `make_classroom_tutoring` | tutoring hours and test scores in classes | continuous treatment; what the focal set does to the estimand |
| `make_homophily_trap` | an online course where the true peer effect is **zero** | a failure mode, and how a negative control catches it |
| `make_toy_graph` | 30 hand-drawn nodes | pictures of focal sets and folds |
| `make_lastfm_promo` | a promotion on the **real** LastFM Asia social network (7,624 users) | real topology and covariates, known truth |

Each returns the data, the true effects and a story card: `print(ds.DESCR)`.

## Check your analysis

```python
from graphdml import placebo_test, negative_control_test, two_hop_test

placebo_test(model, data)                          # shuffled treatment -> no effect?
negative_control_test(model, data, last_year_y)    # outcome T can't affect -> no effect?
two_hop_test(model, data)                          # do friends-of-friends matter?
```

No estimator can rule out unobserved confounding on its own. In `make_homophily_trap`,
GraphDML reports a large spurious peer effect, and `negative_control_test` flags it.

## How graphdml relates to the paper

graphdml follows the paper's model, focal-set idea and cross-fitting. Its defaults differ
where our derivations and benchmarks showed improvements; `GraphDML(mode="paper")`
reproduces the published procedure. Details and evidence are in
[docs/deviations.md](docs/deviations.md). In short:

* **IV-type score by default.** The paper's partialling-out score is biased toward zero when
  the treatment model is wrong. We derive the attenuation factor and confirm it to within 3%.
  The IV-type score stays consistent if either nuisance is right.
* **Buffered nuisance training.** Training only on focal nodes undercovers. With buffered
  training (every node independent of the held-out fold), coverage is 0.93–0.99 across our
  benchmarks.
* **Correct intervals.** Our variance matches statsmodels' HC0 exactly, and matches
  DoubleML when the graph has no edges.

## Validation

Validation follows a five-layer protocol ([VALIDATION.md](VALIDATION.md)) with
pre-specified pass criteria. Current results are in
[docs/validation-results.md](docs/validation-results.md):

* Coverage of 95% intervals across 18 cells (ER, BA and SBM graphs; oracle and learned
  nuisances; continuous and binary treatments): 0.907–0.970 at 300 seeds. The one cell below
  the screening band covered at 0.944 when rerun with 2,000 fresh seeds.
* With default learners on four datasets: relative bias within ±2.4% and coverage 0.93–0.99.
* On the real LastFM Asia network (semi-synthetic, 100 replications): bias within 2.3%, coverage
  0.92–0.96. Network-blind methods are off by 37–71% and never cover.
* **Known limitation:** the focal set keeps only about 10–25% of nodes. An all-node estimator has
  about 2.5× lower error but undercovers. Dependence-robust variance with cluster
  cross-fitting is the top V1.1 item.

```bash
pytest              # correctness suite (~80 tests, ~30 s)
pytest -m slow      # coverage gate
```

## Roadmap

- [x] **M1: core.** Estimator, focal sets, scikit-learn nuisances, diagnostics, story
  datasets, validation suite.
- [ ] **M2: GNN learners.** scikit-learn-compatible GIN, GCN and GraphSAGE (`[gnn]` extra).
- [x] **M3: Neo4j.** Load from Cypher, write estimates and residuals back, GDS embeddings as
  features, Docker demo.
- [ ] **M4: release.** Docs site, tutorials, PyPI.
- [ ] **V1.1.** All-node inference (dependence-robust variance, cluster cross-fitting),
  the small (<1 point) undercoverage seen on hub-heavy graphs, sensitivity analysis for unobserved confounding, and
  semi-synthetic plus experimental real-data benchmarks.

## Citation

If you use graphdml, please cite the paper it builds on:

```bibtex
@inproceedings{khatami2025graph,
  title     = {Graph Machine Learning based Doubly Robust Estimator for Network Causal Effects},
  author    = {Khatami, Seyedeh Baharan and Parikh, Harsh and Chen, Haowei and Roy, Sudeepa and Salimi, Babak},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {258},
  pages     = {4366--4374},
  year      = {2025}
}
```

See also [CITATION.cff](CITATION.cff). graphdml is MIT licensed.
