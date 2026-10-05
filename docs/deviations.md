# Deviations from Khatami et al. (2025) and its reference code

graphdml builds on the method of Khatami, Parikh, Chen, Roy and Salimi (AISTATS 2025) and
its reference implementation ([BaharanKh/GDML](https://github.com/BaharanKh/GDML), MIT
licence). Its defaults differ in the ways listed below, each backed by a derivation, a test
or a benchmark. `GraphDML(mode="paper")` reproduces the published procedure for
replication.


## Defaults

| # | Topic | Paper / reference code | graphdml default | Why | Evidence |
|---|---|---|---|---|---|
| 1 | Focal-set definition | Paper Def. 3.1: open neighborhoods disjoint. Code: pairwise distance ≥ 3 | Disjoint *dependency sets* (distance ≥ 3 for one-hop exposure; generalises to any exposure and to directed graphs) | The open-neighborhood definition admits adjacent nodes whose scores share noise. The code is already right; we generalise it | `tests/test_focal.py` |
| 2 | Focal-set order | Random | Lowest degree first (random available) | Larger focal set, so smaller standard errors | `test_min_degree_strategy_finds_larger_sets_on_average` |
| 3 | Nuisance training data | Other folds' focal nodes only | Every node independent of the held-out scores ("buffered") | Several times more training data, and no covariate shift between focal nodes and the neighbors the treatment model must predict | Focal-only training on referral_app: direct effect +10% biased with coverage 0.80; peer coverage 0.89–0.90 on two datasets; RMSE up to 1.8× higher. With the partialling-out score, peer coverage fell as low as 0.72. Buffered: 0.93–0.99 ([results](validation-results.md#design-choices)) |
| 4 | Score | Partialling-out | IV-type | Partialling-out is attenuated when the treatment model is wrong; IV-type is consistent if either the treatment model or *g* is right | Derivation in [methodology §4](methodology.md#4-scores); `tests/test_robustness.py`; IV-type's RMSE was lower in 7 of 8 dataset × effect cells (up to 3× lower on the paper's DGPs); on referral_app's direct effect, partialling-out was 1% lower |
| 5 | Fold aggregation | DML1 (average of fold estimates) | DML2 (pooled moment) | Standard choice; more stable with small folds | |
| 6 | Number of folds | K = 3 | K = 5 | The paper notes that K = 4 or 5 is more reliable than small K | |
| 7 | Final-stage regression | Code adds an intercept and the covariates to the residual regression | Residuals only (paper eq. 13) | Matches the score whose variance Theorem 4.1 gives | |
| 8 | Variance and intervals | Theorem 4.1 | Theorem 4.1 sandwich (equal to HC0 for partialling-out), implemented and tested directly | | Matches statsmodels HC0 and DoubleML exactly; coverage gate in [validation-results.md](validation-results.md#coverage) |
| 9 | Nuisance features | One-layer GIN (one-hop aggregation) | Own + one- and two-hop aggregates, any scikit-learn model | $\mathbb{E}[Y \mid X, A]$ depends on two-hop covariates (methodology §3) | No measurable difference in our benchmarks; kept because it is never worse |
| 10 | Nuisance learners | GIN, 300 epochs | Regularised histogram gradient boosting; GNNs planned as the `[gnn]` extra | Over-fit propensities add noise that biases partialling-out toward zero | With scikit-learn's default HGB and partialling-out on flu_town (12 seeds): 3.5% of propensities clipped, peer effect 13% too small; regularised: 0% clipped, 8% too small; regularised + IV-type: 1% |
