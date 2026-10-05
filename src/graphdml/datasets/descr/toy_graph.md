# Toy Graph

A hand-drawn 30-node graph for pictures, not estimation: three 8-node clusters, two short
chains linking them, and two isolated nodes. `extras["positions"]` holds a fixed layout.

Use it to see:

* **Dependency sets.** With the one-hop exposure, node *i*'s score involves node *i* and its
  neighbors. Two nodes can both be focal only if those sets do not overlap, i.e. they are at
  least three hops apart.
* **Focal sets.** `select_focal_set` picks a maximal set of such nodes; isolated nodes are
  always focal (they block nothing).
* **Folds.** Focal nodes are split into folds; each held-out fold's dependency sets are
  excluded from training the nuisance models.

```python
import numpy as np
from graphdml.datasets import make_toy_graph
from graphdml.focal import dependency_matrix, select_focal_set

ds = make_toy_graph()
dep = dependency_matrix(30, [ds.data.pattern])
print(select_focal_set(dep, random_state=0))
```

Truth (for completeness): direct effect 1.0, peer effect 0.5 per treated neighbor. With 30 nodes
the focal set has only a handful of members, far too few to estimate anything.
