# graphdml

**Causal effects on networks, with honest confidence intervals.**

When a treatment reaches one person, it often reaches their friends too: a vaccine protects
the neighbors, a coupon gets shared, a training spreads by word of mouth. graphdml estimates

* the **ATE**: the average effect of treating everyone vs no one, including the ripple
* the **direct effect** (ADE): what the treatment does for the person who receives it
* the **peer effect** (APE): what it does for the people connected to them

using double machine learning with any scikit-learn model, adjusting for confounding that
runs through the network, with confidence intervals that account for neighbors being
dependent. Graphs can come from pandas, networkx or Neo4j.

## Install

```bash
pip install graphdml               # core
pip install "graphdml[neo4j]"      # with Neo4j support
```

Requires Python 3.10 or newer.

## Quickstart

```python
from graphdml import GraphDML
from graphdml.datasets import make_flu_town

ds = make_flu_town()      # 5,000 residents; true effects: -2.0 and -0.5 sick days
model = GraphDML(exposure="sum", random_state=0).fit(ds.data)
print(model.summary())
```

## Where to go next

| | |
|---|---|
| [Example datasets](datasets.md) | Eight datasets with known truth, and the story behind each |
| [Methodology](methodology.md) | Exactly what `fit` computes, and why |
| [Design choices](design.md) | Each default, with the evidence behind it |
| [Validation](validation.md) | The protocol, and the [results](validation-results.md) |
| [Neo4j integration](neo4j.md) | Load graphs, write results back, GDS embeddings |
| [API reference](api.md) | Every public class and function |

## Does it find real signals?

```python
import graphdml
print(graphdml.selftest())
```

Simulates a world with planted effects and a null twin with none, and checks that graphdml
finds the first and stays quiet on the second, while a method that ignores the network
does not. See the [README](https://github.com/Samrdreyfuss/graphdml#does-it-find-real-signals-run-the-self-test)
for sample output.

## Citing

The estimator builds on Khatami, Parikh, Chen, Roy and Salimi,
[*Graph Machine Learning based Doubly Robust Estimator for Network Causal Effects*](https://proceedings.mlr.press/v258/khatami25a.html)
(AISTATS 2025). Please cite their paper along with graphdml; see
[CITATION.cff](https://github.com/Samrdreyfuss/graphdml/blob/main/CITATION.cff).
