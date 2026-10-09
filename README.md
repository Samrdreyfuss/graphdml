# graphdml

[![PyPI](https://img.shields.io/pypi/v/graphdml)](https://pypi.org/project/graphdml/)
[![Python](https://img.shields.io/pypi/pyversions/graphdml)](https://pypi.org/project/graphdml/)
[![CI](https://github.com/Samrdreyfuss/graphdml/actions/workflows/ci.yml/badge.svg)](https://github.com/Samrdreyfuss/graphdml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Samrdreyfuss/graphdml/blob/main/LICENSE)

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

## Installation

graphdml is on [PyPI](https://pypi.org/project/graphdml/) and needs **Python 3.10 or newer**.

```bash
pip install graphdml               # core: numpy, scipy, pandas, scikit-learn
pip install "graphdml[neo4j]"      # + Neo4j integration (official neo4j driver)
```

Check the installation:

```bash
python -c "import graphdml; print(graphdml.__version__)"
```

Upgrade with `pip install --upgrade graphdml`. With [uv](https://docs.astral.sh/uv/), use
`uv pip install graphdml` or `uv add graphdml`.

<details>
<summary>Trouble installing?</summary>

* **`pip: command not found`**: use `python3 -m pip install graphdml`.
* **"No matching distribution found"**: your Python is older than 3.10 (macOS ships 3.9).
  Install a newer Python, e.g. `brew install python@3.12` or `uv python install 3.12`.
* **"externally-managed-environment"**: your Python (e.g. Homebrew's) protects its own
  packages. Install into a virtual environment:
  `python3.12 -m venv .venv && source .venv/bin/activate && pip install graphdml`.
* **Latest development version**: `pip install "git+https://github.com/Samrdreyfuss/graphdml.git"`.

</details>

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
[Neo4j guide](https://github.com/Samrdreyfuss/graphdml/blob/main/docs/neo4j.md) covers directed graphs, custom Cypher and GDS embeddings.

## Example datasets

All of these are included in the package. Five generate their data offline; two need data
from outside: one downloads automatically, one you download yourself.

| Dataset | Story | Shows | Data needed |
|---|---|---|---|
| `make_flu_town` | flu shots and sick days among neighbors | direct vs peer effects, network confounding | none (generated offline) |
| `make_referral_app` | promo coupons in an app with influencers | choosing the exposure map | none (generated offline) |
| `make_classroom_tutoring` | tutoring hours and test scores | continuous treatment, who the estimate is about | none (generated offline) |
| `make_homophily_trap` | a course whose true peer effect is zero | a failure mode and how to catch it | none (generated offline) |
| `make_toy_graph` | 30 hand-drawn nodes | pictures of focal sets and folds (too small to estimate anything) | none (generated offline) |
| `make_lastfm_promo` | a promotion on the real LastFM Asia network | real topology and covariates, known truth | **internet on first use** (6.5 MB, automatic) |
| `load_insurance_experiment` | a randomized field experiment with 4,902 farmers | real data, no known truth | **you download it** (free account) |

The first five are simulated with known true effects, so you can check an estimate against
the answer. Every dataset comes with a story card: `print(ds.DESCR)`.

### Datasets that need outside data

**LastFM Asia** (`make_lastfm_promo`): the social network and covariates are real, the
treatment and outcome are simulated. On first use it downloads the network from
[SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html), checks its checksum and
caches it in `~/.cache/graphdml` (set `GRAPHDML_DATA` to change the location). After that it
works offline. If you are offline or behind a firewall, download `lastfm_asia.zip` yourself
and put it in the cache folder. Please cite Rozemberczki and Sarkar (CIKM 2020), also shown
in `ds.extras["citation"]`.

**Insurance experiment** (`load_insurance_experiment`): a real randomized experiment with a
real friendship network, from Cai, de Janvry and Sadoulet (2015). graphdml does not
redistribute it. Download the replication package from
[openICPSR project 113593](https://www.openicpsr.org/openicpsr/project/113593/version/V1/view)
(free account, CC BY 4.0 license), unzip it, and point the loader at the folder that holds
`0422survey.dta` and `0422allinforawnet.dta`:

```python
from graphdml.datasets import load_insurance_experiment

ds = load_insurance_experiment("~/Downloads/113593-V1/data/data")
```

If the files are not found, the error tells you where to get them. See the dataset's story
card for the published estimates to compare against.

## Does it find real signals? Run the self-test

graphdml ships a check you can run on your own machine. It simulates two worlds with the
same strong network confounding: one with planted effects (direct +1.0, peer +0.6) and a
**null twin** with none. It fits the model many times and compares with the known truth.

```python
import graphdml
print(graphdml.selftest(n_reps=60))     # about 30 seconds on a 10-core laptop
```

```
graphdml self-test: PASSED (60 repetitions per world)

                truth  mean_estimate  coverage  detected  naive_detected
world   effect
planted direct  1.000          1.049     0.917     1.000           1.000
        peer    0.600          0.635     0.983     1.000           1.000
null    direct  0.000          0.050     0.917     0.083           1.000
        peer    0.000          0.036     0.967     0.033           1.000
```

In the planted world graphdml finds both effects every time. In the null world, where
nothing is happening, it raises a false alarm 3 to 8% of the time (5% is the target), while
a method that ignores the network reports an effect **every time**, because the
confounding looks like a treatment effect. The full output also lists the pass criteria.
You can test your own settings with `graphdml.selftest(estimator=GraphDML(...))`, and use
the data directly with `make_signal_check()`.

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

Details: [methodology](https://github.com/Samrdreyfuss/graphdml/blob/main/docs/methodology.md) · [design choices and evidence](https://github.com/Samrdreyfuss/graphdml/blob/main/docs/design.md).

## Validation

graphdml is tested against pre-specified criteria ([protocol](https://github.com/Samrdreyfuss/graphdml/blob/main/docs/validation.md),
[results](https://github.com/Samrdreyfuss/graphdml/blob/main/docs/validation-results.md)):

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
(AISTATS 2025). Please cite their paper along with graphdml ([CITATION.cff](https://github.com/Samrdreyfuss/graphdml/blob/main/CITATION.cff)).

## Contributing and changes

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/Samrdreyfuss/graphdml/blob/main/CONTRIBUTING.md). Release
notes are in [CHANGELOG.md](https://github.com/Samrdreyfuss/graphdml/blob/main/CHANGELOG.md).

MIT licensed.
