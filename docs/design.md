# Design choices

Each default below was chosen on the basis of a derivation, a test or a benchmark.
`GraphDML(mode="original")` switches all of them to the procedure of the original method,
which is useful for comparisons and replication.

| Choice | Default | Original procedure | Why | Evidence |
|---|---|---|---|---|
| Focal set | nodes with disjoint *dependency sets* (pairwise distance ≥ 3 for one-hop exposure; generalises to any exposure and to directed graphs) | same rule, one-hop exposure only | scores are independent exactly when their noise sets do not overlap | `tests/test_focal.py` |
| Focal-set order | lowest degree first | random | larger focal set, so smaller standard errors | `test_min_degree_strategy_finds_larger_sets_on_average` |
| Nuisance training data | every node independent of the held-out scores ("buffered") | focal nodes of the other folds | several times more data, and no covariate shift between focal nodes and the neighbors the treatment model predicts for | focal-only training: direct effect +10% biased with coverage 0.80 on `referral_app`, peer coverage down to 0.72 with partialling-out; buffered: 0.93–0.99 ([results](validation-results.md#design-choices)) |
| Score | IV-type | partialling-out | partialling-out is attenuated when the treatment model is wrong; IV-type is consistent if either the treatment model or *g* is right | [methodology §4](methodology.md#4-scores), `tests/test_robustness.py`; lower RMSE in 7 of 8 benchmark cells (up to 3×) |
| Fold aggregation | pooled moment (DML2) | average of fold estimates (DML1) | more stable with small folds | |
| Folds | 5 | 3 | more training data per fold | |
| Features | own covariates + one- and two-hop neighbor aggregates | own + summed one-hop neighbor covariates | E[Y \| X, A] depends on two-hop covariates ([methodology §3](methodology.md#3-identification)) | no measurable difference in benchmarks; never worse |
| Nuisance learners | regularised histogram gradient boosting | graph neural network | over-fit propensities add noise that biases partialling-out toward zero | `flu_town`, 12 seeds: library-default boosting 13% attenuation, regularised 8%, regularised + IV-type 1% |
| Variance | sandwich (HC0-equivalent for partialling-out); optional cluster-robust | sandwich | | matches statsmodels and DoubleML exactly; [coverage gate](validation-results.md#coverage) |
