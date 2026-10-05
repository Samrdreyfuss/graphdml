# Homophily Trap

**Question.** Does taking an online course raise salary growth, and does it rub off on friends?

**Teaches.** A failure mode: when an *unobserved* trait drives both friendships and outcomes,
GraphDML (like every method that assumes no unobserved confounding) finds a peer effect that
does not exist. And how to catch it.

| | |
|---|---|
| Nodes | 5,000 professionals |
| Edges | friendships formed mostly between people with similar (unobserved) ambition |
| Treatment | `took_course` (0/1) |
| Outcome | `salary_growth` (%) |
| Covariates | `age`, `education_years` |
| Exposure map | `"mean"`: share of friends who took the course |
| True direct effect | **+1.0** percentage points |
| True peer effect | **0**: friends' courses do nothing |
| Hidden | `extras["latent_ambition"]`, never shown to the estimator |
| Negative control | `extras["negative_control"]`: *last year's* salary growth |

**What goes wrong.** Ambitious people take the course, earn raises, and befriend other ambitious
people. Friends' enrolment is therefore a proxy for your own ambition, so it "predicts" your
salary growth: GraphDML reports a large, confident, and entirely spurious peer effect, and an
inflated direct effect.

**How you would know.** Last year's salary growth shares the confounder (ambition) but cannot
have been caused by this year's course. Run the analysis on it as if it were the outcome; any
"effect" is confounding:

```python
from graphdml import GraphDML, negative_control_test
from graphdml.datasets import make_homophily_trap

ds = make_homophily_trap()
model = GraphDML(exposure=ds.exposure, random_state=0)
print(model.fit(ds.data).summary_frame())          # peer effect looks real ...
print(negative_control_test(model, ds.data, ds.extras["negative_control"]))  # ... FLAG
```

Note that `two_hop_test` does *not* catch this case: homophily makes friends-of-friends weaker
proxies than friends, so the two-hop coefficient is small. Use negative controls and domain
knowledge about how ties form; no estimator can rule out unobserved confounding on its own.
