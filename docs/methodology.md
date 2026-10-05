# Methodology

This document specifies exactly what `GraphDML.fit` computes; the tests in `tests/` check
the statements made here. The approach builds on Khatami et al. (AISTATS 2025); defaults
that differ from their procedure are listed, with evidence, in [design.md](design.md).

## 1. Setup

A network of $n$ nodes with adjacency $A$, where $A_{ij} \neq 0$ means node $j$ can
influence node $i$ (for undirected graphs $A$ is symmetric). Each node has pre-treatment
covariates $X_i$, a treatment $T_i$ (binary or continuous) and an outcome $Y_i$.

## 2. Model, estimands and assumptions

$$
\begin{aligned}
T &= m_0(X, A) + \varepsilon_T, & \mathbb{E}[\varepsilon_T \mid X] &= 0,\\
Y &= \theta_0\, T + \alpha_0\, E T + g_0(X, A) + \varepsilon_Y, & \mathbb{E}[\varepsilon_Y \mid X, T] &= 0.
\end{aligned}
$$

* $E$ is a known, **linear exposure map**: $Z = ET$ summarises others' treatments
  (`"sum"`: $E = A$; `"mean"`: $E = D^{-1}A$; weighted, two-hop or custom operators).
  $E$ has a zero diagonal: a node's own treatment enters through $\theta$.
* $m_0, g_0$ are unknown. They may depend on a node's own *and its neighbors'* covariates;
  this is **network confounding**.
* $\theta_0$ is the **average direct effect** (ADE) and $\alpha_0$ the **average peer
  effect** (APE). Several exposures give one $\alpha$ per exposure.

**Assumptions.** (A.1) Noise terms are independent across nodes given
covariates. (A.2) Interference is confined to the exposure neighborhood. (A.3) The exposure
map is correct. (A.4) Positivity of treatment and exposure. (A.5) Consistency. (A.6) No
unobserved confounding given $X$ and $A$. A.2, A.3 and A.6 cannot be verified from data;
`two_hop_test` and `negative_control_test` can falsify some violations.

**Under heterogeneous effects.** If $\theta$ varies across nodes, the partially linear model
is misspecified and the estimate is a weighted average of node-level effects (weights
proportional to the conditional variance of the treatment), taken over the *focal set*
(section 5), not over all nodes. `GraphDataset.focal_truth` makes this visible in
`make_classroom_tutoring(heterogeneous=True)`.

## 3. Identification

Let $\ell_0(X, A) = \mathbb{E}[Y \mid X, A]$. Because $E$ is linear,
$\mathbb{E}[ET \mid X, A] = E\, m_0$, so

$$
Y - \ell_0 = \theta_0 (T - m_0) + \alpha_0\, E(T - m_0) + \varepsilon_Y .
$$

The **peer residual** is the exposure map applied to *all nodes'* treatment residuals,
$E(T - m_0)$. In particular it needs treatment-model predictions at a focal node's
neighbors, which are usually not focal themselves (section 6).

**Two-hop dependence of $\ell_0$.** $\ell_0 = \theta_0 m_0 + \alpha_0 E m_0 +
g_0$. Even with one-hop interference and one-hop confounding, $E m_0$ depends on
covariates two hops away. The default featurizer therefore includes two-hop aggregates.

## 4. Scores

Every score is linear in $\zeta = (\theta, \alpha)$: $\psi_i(\zeta) = R_i\,(y_i - D_i^\top
\zeta)$, with residualised regressors $R_i = (T_i - \hat m_i,\ [E(T - \hat m)]_i)$.

| score | $D_i$ | $y_i$ | consistent if |
|---|---|---|---|
| `partialling-out` | $R_i$ | $Y_i - \hat\ell_i$ | $\hat m$ correct |
| `iv-type` (default) | $(T_i, Z_i)$ | $Y_i - \hat g_i$ | $\hat m$ **or** $\hat g$ correct |

**Robustness to nuisance errors.** Write $V = T - m_0$ and suppose the
treatment model is off by $\delta = m_0 - \hat m$ while $\hat\ell = \ell_0$ is exact. With a
single regressor, the partialling-out estimate converges to

$$
\theta_0\,\frac{\mathbb{E}[V^2]}{\mathbb{E}[V^2] + \mathbb{E}[\delta^2]},
$$

an attenuation toward zero (the cross terms vanish because $\delta$ is a function of
covariates and $\mathbb{E}[V \mid X] = 0$). If instead $\hat m = m_0$ and $\hat \ell$ is
wrong, $\mathbb{E}[(\ell_0 - \hat\ell) V] = 0$ and the estimate is consistent. So the
partialling-out score is robust to the outcome model but **not** to the treatment model; it
is Neyman-orthogonal, so the bias is second order, but it is not doubly robust in the
classical sense. The IV-type score uses $D = (T, Z)$ and
$\hat g = $ a regression of $Y - \tilde\theta T - \tilde\alpha Z$ on features (with
$\tilde\zeta$ a preliminary partialling-out estimate, as in DoubleML). It is consistent if
either $\hat m$ or $\hat g$ is correct. `tests/test_robustness.py` checks all four cases
with oracle and constant nuisances, and checks the attenuation formula to within 3%.

In practice machine-learned propensities are noisy, and the noise acts like $\delta$. With
the default learners, partialling-out estimates were attenuated by up to 6%, while IV-type
estimates stayed within ±2.4% of the truth (see [validation-results.md](validation-results.md)).

## 5. Dependency sets and focal sets

Node $i$'s score involves the noise of node $i$ and of every node in the support of row $i$
of each exposure operator. Its **dependency set** is
$S_i = \{i\} \cup \bigcup_k \operatorname{supp}(E_{k,i\cdot})$. If $S_i \cap S_j =
\emptyset$, the scores of $i$ and $j$ are independent given covariates.

A **focal set** is a maximal set of nodes with pairwise-disjoint dependency sets, built
greedily (`select_focal_set`): visit nodes in some order and accept a node if its dependency
set touches no accepted node's set.

* For a one-hop exposure on an undirected graph, this means pairwise graph distance at
  least 3. Requiring only *open* neighborhoods to be disjoint would not be enough: two
  adjacent nodes with no common neighbor would qualify, yet their scores share noise.
* With an added two-hop exposure (`two_hop_test`) the requirement becomes distance at least 5.
* For directed graphs, dependency sets follow in-neighbors.
* Order: `"min_degree"` (default) visits low-degree nodes first and yields larger focal
  sets. `"random"` gives a less degree-skewed set.

The focal set size $n_f$ is the **effective sample size**: standard errors shrink like
$1/\sqrt{n_f}$, not $1/\sqrt{n}$.

## 6. Cross-fitting

Focal nodes are split into $K$ folds (`KFold`, shuffled). For held-out fold $k$ with focal
nodes $I_k$:

1. $U_k = \bigcup_{i \in I_k} S_i$ is the set of nodes whose noise enters the held-out scores.
2. Training sets:
   * `nuisance_training="focal"`: the focal nodes of the other folds.
   * `nuisance_training="buffered"` (default): the treatment model trains on every node
     outside $U_k$; the outcome model trains on every node $u$ with $S_u \cap U_k =
     \emptyset$. Both training sets are independent of the held-out scores, as
     cross-fitting requires, and are typically several times larger than the focal training
     set.
3. The treatment model predicts $\hat m$ on all of $U_k$ (focal nodes *and* their
   neighbors). This gives $\hat V = T - \hat m$ on $U_k$ and the peer residual
   $[E \hat V]_i$ for $i \in I_k$. Because dependency sets of focal nodes are disjoint, each
   non-focal neighbor belongs to exactly one held-out fold, and its prediction comes from a
   model that never saw it.
4. The outcome model predicts $\hat\ell_i$ (or $\hat g_i$ for IV-type) for $i \in I_k$.

**Why buffered training matters.** Training only on focal nodes uses less
data. It also creates a covariate shift: focal nodes skew toward low degree, but the treatment
model must predict at their (higher-degree) neighbors. Errors in $\hat m$ at neighbors
attenuate the peer effect (section 4). In benchmarks, focal-only training undercovers
(peer coverage as low as 0.72 with the partialling-out score; direct-effect coverage 0.80 on
`referral_app` with the IV-type score), while buffered training stays at 0.93–0.99.

## 7. Features for the nuisance models

Nuisances are fit by any scikit-learn estimator on a feature matrix built once from
$(X, A)$, never from $T$ or $Y$, so featurising the whole graph does not leak. The default
`NeighborhoodFeatures(aggs=("mean","max","min"), hops=2)` returns own covariates, one-hop
mean/max/min aggregates, the neighbor mean of those (two hops), and in-degree.
`PrecomputedFeatures` accepts any covariate-only embedding (e.g. Neo4j GDS FastRP).
Binary treatments use a classifier's `predict_proba`, clipped to `[0.01, 0.99]`.

## 8. Final stage and inference

* `aggregation="dml2"` (default) solves the pooled moment $\sum_i \psi_i(\zeta) = 0$ over all
  focal nodes. `"dml1"` averages per-fold solutions.
* Variance: $\hat\Sigma = \hat J^{-1} \hat\Omega \hat J^{-\top} / n_f$ with
  $\hat J = \frac{1}{n_f}\sum_i R_i D_i^\top$ and $\hat\Omega = \frac{1}{n_f}\sum_i \hat u_i^2
  R_i R_i^\top$. For partialling-out this is exactly the HC0 covariance of the OLS of
  $Y - \hat\ell$ on the residuals (tested against statsmodels).
* Normal-approximation intervals $\hat\zeta \pm z_{1-a/2}\, \hat\sigma$.
* `n_rep > 1` repeats focal-set selection and fold splitting, takes the median estimate, and
  inflates the variance by the spread across repetitions (Chernozhukov et al. 2018, §3.4).

## 9. Diagnostics and falsification

`summary()` reports out-of-fold nuisance quality, propensity range and clipping, focal vs
overall degree, the share of focal nodes without neighbors, residual collinearity, and
residual balance (max $|\mathrm{corr}(\hat V, \text{feature})|$ against a $3/\sqrt{n_f}$
noise level). It warns on small focal sets, weak overlap, collinearity and isolated focal
nodes. Falsification tests (`graphdml.diagnostics`):

* `placebo_test`: permuted treatments must show no effect (tests the pipeline).
* `negative_control_test`: an outcome the treatment cannot affect must show no effect
  (tests for unobserved confounding such as homophily).
* `two_hop_test`: adds a two-hop exposure; a non-zero coefficient contradicts A.2.

## 10. Original procedure

`GraphDML(mode="original")` reproduces the original GDML procedure: random-order focal set,
focal-only nuisance training, partialling-out score, DML1, $K = 3$, and own plus summed-neighbor
features (the inputs of a one-layer GIN). The nuisance learners are whatever you pass; a
GIN learner arrives with the `[gnn]` extra.

## 11. Out of scope for V1

Heterogeneous effects $\theta(X)$, nonlinear exposure maps, interference beyond the exposure
neighborhood, outcome contagion ($Y_j \to Y_i$), sensitivity analysis for unobserved
confounding, and dependence-robust inference without a focal set (e.g. network HAC).
