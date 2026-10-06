# graphdml

**Causal effects on networks, with honest confidence intervals.**

When a treatment reaches one person, it often reaches their friends too: a vaccine protects
the neighbors, a coupon gets shared, a training spreads by word of mouth. graphdml estimates:

* the **ATE**: the average effect of treating everyone vs no one, including the ripple
* the **direct effect** (ADE): what the treatment does for the person who receives it
* the **peer effect** (APE): what it does for the people connected to them

It uses double machine learning with any scikit-learn model, adjusts for confounding that
runs through the network, and reports confidence intervals that account for neighbors
being dependent. Graphs can come from pandas, networkx or Neo4j, and results can be written
back to Neo4j.

## Install

```bash
pip install graphdml            # core
pip install "graphdml[neo4j]"   # with Neo4j support
```

Requires Python 3.10+. Core dependencies: numpy, scipy, pandas, scikit-learn.

## Quickstart

```python
from graphdml import GraphDML
from graphdml.datasets import make_flu_town

ds = make_flu_town()      # 5,000 residents; true effects: -2.0 and -0.5 sick days
model = GraphDML(exposure="sum", random_state=0).fit(ds.data)
print(model.summary())
```

```
                       coef   std err        z    P>|z|      [2.5%    97.5%]
ATE (total)          -4.437    0.4217   -10.52    0.000     -5.263     -3.61
direct (ADE)         -1.879    0.1275   -14.73    0.000     -2.129    -1.629
peer (APE): sum     -0.4341    0.0689    -6.30    0.000    -0.5692   -0.2991

  ATE (total): change in Y from treating every node vs none (direct + peer)
  direct (ADE): change in Y when own treatment goes 0 → 1, neighbors unchanged
  peer (APE): sum: change in Y per additional treated neighbor
```

Vaccinating the whole town would cut sick days by about 4.4 (truth: 4.9): 1.9 from each
person's own shot, plus 0.4 for each of their roughly 6 vaccinated neighbors. The full
summary also reports diagnostics, warnings and the assumptions the estimate relies on.

**Coming from DML?** The `ATE (total)` row is the number you are used to: the average effect
of treatment vs no treatment for everyone. On a network it splits into a direct part and a
peer part, and the peer part is what standard DML misses. Access them with `model.ate_`,
`model.ade_` and `model.ape_`, or as a DataFrame with `model.summary_frame()`. Methods that ignore the network get this example wrong, because health-conscious
residents cluster together:

```python
from graphdml.baselines import compare_methods
compare_methods(ds.data, exposure="sum", truth=ds.truth)
```

| Method | Effect | Estimate | 95% CI | Truth |
|---|---|---|---|---|
| OLS, no network | direct | −2.39 | [−2.52, −2.26] | −2.0 ✗ |
| OLS + exposure, no neighbor covariates | peer | −0.66 | [−0.68, −0.63] | −0.5 ✗ |
| DML, own covariates only | direct | −2.41 | [−2.54, −2.28] | −2.0 ✗ |
| **GraphDML** | direct | **−1.88** | [−2.13, −1.63] | −2.0 ✓ |
| **GraphDML** | peer | **−0.43** | [−0.57, −0.30] | −0.5 ✓ |

## Your own data

```python
from graphdml import GraphData, GraphDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

data = GraphData.from_pandas(
    nodes, edges,                               # edges: source influences target
    features=["age", "income"], treatment="treated", outcome="spend", node_id="user_id",
)
model = GraphDML(
    model_y=RandomForestRegressor(min_samples_leaf=20),     # any scikit-learn model
    model_t=RandomForestClassifier(min_samples_leaf=20),
    exposure="mean",                                         # share of treated neighbors
).fit(data)
model.summary_frame()                                        # estimates as a DataFrame
```

`GraphData.from_edges` and `GraphData.from_networkx` work too.

### Neo4j

```python
from graphdml.io.neo4j import Neo4jGraphSource, Neo4jResultWriter

src = Neo4jGraphSource(uri="bolt://localhost:7687", auth=("neo4j", "..."))
data = src.load("Customer", "FRIENDS_WITH", features=["age", "tenure"],
                treatment="got_coupon", outcome="spend", id_property="customer_id")
model = GraphDML(exposure="mean").fit(data)
Neo4jResultWriter(driver=src.driver).write(
    model, run_name="coupon-q3", node_label="Customer", id_property="customer_id")
```

Each run is stored as a `(:GDMLRun)` node linked to the nodes it was estimated on, with
per-node predictions and residuals, and never modifies your data. Try it locally with
`docker compose up -d && python -m graphdml.demo --password graphdml-demo`. The
[Neo4j guide](docs/neo4j.md) covers directed graphs, custom Cypher and GDS embeddings.

## Example datasets

| Dataset | Story | Shows |
|---|---|---|
| `make_flu_town` | flu shots and sick days among neighbors | direct vs peer effects, network confounding |
| `make_referral_app` | promo coupons in an app with influencers | choosing the exposure map |
| `make_classroom_tutoring` | tutoring hours and test scores | continuous treatment, who the estimate is about |
| `make_homophily_trap` | a course whose true peer effect is zero | a failure mode and how to catch it |
| `make_lastfm_promo` | a promotion on the real LastFM Asia network | real topology and covariates, known truth |
| `load_insurance_experiment` | a randomized field experiment with 4,902 farmers | real data (download from openICPSR) |

Every dataset comes with a story card: `print(ds.DESCR)`.

## Checking an analysis

```python
from graphdml import negative_control_test, placebo_test, two_hop_test

placebo_test(model, data)                         # shuffled treatment: no effect?
negative_control_test(model, data, last_year_y)   # an outcome T can't affect: no effect?
two_hop_test(model, data)                         # do friends-of-friends matter?
```

No estimator can rule out unobserved confounding on its own; these checks flag common ways
an analysis goes wrong.

## How it works

1. **Features.** Each node's covariates are combined with aggregates of its neighbors'
   covariates, so the models can learn confounding that runs through the network.
2. **Focal set.** Estimates use nodes whose neighborhoods do not overlap, so their
   contributions are independent. This set is the effective sample size.
3. **Cross-fitting.** Nuisance models are trained on all data that is independent of the
   held-out nodes, then used to residualize treatment, peer exposure and outcome.
4. **Final stage.** A robust moment condition gives the direct and peer effects, with
   sandwich (optionally cluster-robust) standard errors.

Details: [methodology](docs/methodology.md) · [design choices and evidence](docs/design.md).

## Validation

graphdml is tested against pre-specified criteria ([protocol](docs/validation.md),
[results](docs/validation-results.md)):

* Matches DoubleML and statsmodels exactly in the cases where they apply.
* 95% intervals cover at the nominal rate on simulated graphs with known truth.
* On the real LastFM Asia network (100 replications): bias under 2.5%, coverage 0.92–0.96.
  Network-blind methods were off by 37–71% and never covered.
* On the randomized insurance experiment, the direct effect matches the published estimate
  (0.148 vs 0.141).

## Status

Alpha. Planned next: graph neural network nuisance models, dependence-robust inference that
uses every node, and sensitivity analysis for unobserved confounding.

## Acknowledgements and citation

The estimator builds on Khatami, Parikh, Chen, Roy and Salimi, [*Graph Machine Learning based
Doubly Robust Estimator for Network Causal Effects*](https://proceedings.mlr.press/v258/khatami25a.html)
(AISTATS 2025). Please cite their paper along with graphdml ([CITATION.cff](CITATION.cff)).

MIT licensed.
