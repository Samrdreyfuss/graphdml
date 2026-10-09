# Classroom Tutoring

**Question.** How much does an hour of tutoring raise a student's test score, and do students
benefit when their friends get tutored?

**Teaches.** Continuous treatments, community structure, and what the focal set does to the
*estimand* when effects differ across students.

| | |
|---|---|
| Nodes | 5,000 students in 200 classes of 25 |
| Edges | friendships, almost all within a class; higher-GPA students have more friends |
| Treatment | `tutoring_hours` per week (continuous, >= 0) |
| Outcome | `test_score` |
| Covariates | `prior_gpa`, `parent_education` (0-3), `free_lunch` |
| Exposure map | `"mean"`: friends' average tutoring hours |
| True direct effect | **+1.5** points per hour of tutoring |
| True peer effect | **+0.4** points per hour of friends' average tutoring |

**What confounds.** Struggling students get more tutoring; so do students whose friends have
well-educated parents, and friends' prior GPA also lifts scores. Naive regression gets even the
*sign* of the peer effect wrong.

**The focal set and the estimand.** GraphDML estimates effects from a *focal set* of students
whose neighborhoods do not overlap; it favours students with fewer friends, who here have lower
GPAs. With `heterogeneous=True`, tutoring helps lower-GPA students more, so the effect among
focal students is larger than the school-wide average. Compare:

```python
from graphdml import GraphDML
from graphdml.datasets import make_classroom_tutoring

ds = make_classroom_tutoring(heterogeneous=True)
m = GraphDML(exposure=ds.exposure, random_state=0).fit(ds.data)
print("school-wide truth:", ds.truth["ade"])
print("focal-set truth:  ", ds.focal_truth(m.focal_nodes_)["ade"])
print("estimate:         ", m.ade_)
```

With constant effects (the default) both truths coincide. With varying effects, report the
estimand honestly: "the average effect among students like those in the focal set", and check
how the focal set differs from everyone (`m.diagnostics_["mean_degree_focal"]`).

**Known accuracy on this dataset.** Over 200 draws, estimates run slightly toward zero
(direct about 1.5%, peer about 10%, total about 3%) and 95% intervals cover the truth about
91-92% of the time, a little under nominal. The signs and sizes are right, and ignoring the
network is far worse: it gets the sign of the peer effect wrong.
