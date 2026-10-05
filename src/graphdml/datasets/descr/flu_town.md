# Flu Town

**Question.** Does a flu shot reduce your sick days, and do your vaccinated neighbors protect
you too?

**Teaches.** The difference between a *direct* and a *peer* effect, and why adjusting only for
your own characteristics is not enough when confounders sit in your neighborhood.

| | |
|---|---|
| Nodes | 5,000 residents placed at random in a square town |
| Edges | residents who live close to each other (random geometric graph, ~6 neighbors each) |
| Treatment | `flu_shot` (0/1) |
| Outcome | `sick_days` this season |
| Covariates | `age`, `chronic_condition`, `health_consciousness` |
| Exposure map | `"sum"`: the number of vaccinated neighbors |
| True direct effect | **-2.0** sick days for getting the shot |
| True peer effect | **-0.5** sick days per vaccinated neighbor (herd protection) |

**What confounds.** Health-conscious residents cluster together. The health-consciousness of
your *neighbors* makes you more likely to get vaccinated (social norms) and also lowers your
sick days (a healthier environment). Your own covariates do not capture it.

**What naive methods get wrong.** Regressing sick days on your own shot and your own
covariates credits the shot with the neighborhood's healthiness: it reports roughly -2.4 days
of protection instead of -2.0, and it has no way to report herd protection at all.

```python
from graphdml import GraphDML
from graphdml.datasets import make_flu_town
from graphdml.baselines import compare_methods

ds = make_flu_town()
model = GraphDML(exposure=ds.exposure, random_state=0).fit(ds.data)
print(model.summary())
print(compare_methods(ds.data, exposure="sum", truth=ds.truth))
```

**Knobs.** `confounding=0` removes the neighborhood confounding (naive methods then work for the
direct effect); `avg_neighbors` changes density, and with it the size of the focal set.
