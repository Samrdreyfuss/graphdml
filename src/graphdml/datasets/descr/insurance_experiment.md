# Insurance Experiment (real randomized experiment, real network)

**Question.** Does an intensive information session make rice farmers more likely to buy
weather insurance, and does it spill over to their friends?

**Source.** Cai, de Janvry & Sadoulet (2015), *Social Networks and the Decision to Insure*,
AEJ: Applied. Replication data: openICPSR project 113593 (CC BY 4.0; free account needed).
`load_insurance_experiment(path)` reads the downloaded `.dta` files; graphdml does not ship
the data.

| | |
|---|---|
| Nodes | 4,902 surveyed households in 47 administrative (173 natural) villages |
| Edges | ~17,000 friend nominations (up to five per household), directed: named friend → nominator |
| Treatment | `intensive_session`: randomized assignment to an intensive (vs simple) session |
| Design | two session rounds three days apart (randomized); second-round households were randomly given information about first-round take-up, or none |
| Outcome | `bought_insurance` (0/1) |
| Covariates | household head and farm characteristics, design indicators, village dummies |
| Exposure | share of named friends who attended a first-round intensive session (`extras["exposure"]`) |

**Published estimates (Table 2).** Direct effect of an intensive session on first-round
households: **0.141** (SE 0.026). Peer effect for second-round households without
information: **0.291** (SE 0.082) going from 0% to 100% of friends in first-round intensive
sessions.

**Exposure coding.** The study's exposure counts every named friend in the denominator, and
in the numerator counts friends recorded as first-round intensive participants. That
includes 271 nominations of friends who are not among the surveyed households. Those friends
are not nodes here, so the loader's exposure counts surveyed friends only. The two agree for
97% of households; `extras["published_exposure"]` holds the study's variable for comparison.

**What to try.**

```python
from graphdml import GraphDML
from graphdml.datasets import load_insurance_experiment

ds = load_insurance_experiment("path/to/113593-V1/data/data")
ex = ds.extras
first_round = ~ex["second_round"]
m = GraphDML(exposure=None, estimation_nodes=first_round, random_state=0).fit(ds.data)
print(m.summary_frame())                                  # direct effect
peer_pop = ex["second_round"] & ex["info_none"]
m = GraphDML(exposure=ex["exposure"], estimation_nodes=peer_pop, random_state=0).fit(ds.data)
print(m.cluster_summary_frame(ex["natural_village"]))     # SEs clustered by village
```

Treatment was randomized, so the propensity model is correct by design and both scores are
consistent. Clustered standard errors allow for village-level common shocks, as in the
original study.
