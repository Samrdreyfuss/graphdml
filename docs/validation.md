# Validation protocol

Causal estimates are only useful if they are right, and a wrong number with a confident
interval is worse than no number. graphdml is validated in five layers. Each layer answers a
different question, and each has a fixed pass criterion written down before the evidence is
collected. Results for the current version are in
[validation-results.md](validation-results.md).

| Layer | Question | Where |
|---|---|---|
| 1. Specification | Is the estimator precisely defined? | [methodology.md](methodology.md) |
| 2. Correctness | Does the code compute that estimator? | `pytest` (fast suite) |
| 3. Statistical validity | Does it behave as the theory says? | `pytest -m slow`, `benchmarks/` |
| 4. Realism | Does it hold up on realistic data? | story datasets; real-data benchmarks (planned) |
| 5. User-facing checks | Can users tell when *their* analysis is unreliable? | `summary()` diagnostics, `graphdml.diagnostics` |

## Layer 2: correctness (every commit)

* **Reductions to independent implementations.**
  * With an edgeless graph and no exposure, estimates and standard errors match DoubleML's
    `DoubleMLPLR` (both scores, same folds) to 1e-8.
  * The partialling-out covariance equals statsmodels' HC0 covariance of the final
    regression.
* **Oracle nuisances.** Plugging in the true nuisances recovers the truth.
* **Metamorphic properties.**
  * Scaling Y scales the effects.
  * Shifting T leaves them unchanged.
  * Adding a function of X to Y is absorbed.
  * Two disjoint copies of a graph give the same estimate with SE / √2.
  * Relabelling nodes changes nothing.
* **Focal sets.** Dependency sets are pairwise disjoint and maximal; for one-hop exposure,
  pairwise distance ≥ 3, and with a two-hop exposure, distance ≥ 5. Checked across ER, BA and
  SBM graphs.
* **Robustness claims.** Each score's consistency or bias under deliberately wrong
  nuisances matches the derivation in methodology §4, including the attenuation factor.

## Layer 3: statistical validity

### Coverage criterion

For each cell (DGP × nuisance type × topology × effect), nominal 95% intervals must cover at
95% up to Monte Carlo error:

1. **Screen**: 300 seeds; pass if coverage ∈ 0.95 ± 3·√(0.95·0.05/300) = [0.912, 0.988].
2. **Confirm**: any cell outside the screening band is rerun with 2,000 *fresh* seeds; it
   passes if coverage ∈ 0.95 ± 3·√(0.95·0.05/2000) = [0.935, 0.965].

The release gate requires every cell to pass at stage 1 or stage 2. Seeds are fixed, so
results are reproducible.

**Revision log.** Both changes below were made *after* seeing results, so we record them in
full.

1. The original criterion was coverage ∈ [0.925, 0.975] at 300 seeds. That band is about
   ±2.3 Monte Carlo SEs per cell, so with 18 cells several false failures are expected. The
   first run (partialling-out default) had three cells at 0.917–0.920. Rerunning the ER
   oracle cell with 2,000 fresh seeds gave 0.942 (SE/SD 0.98).
2. We widened the screening band to ±3 Monte Carlo SEs. The next run (IV-type default)
   flagged one cell: BA, linear nuisances, peer effect, at 0.907. Its 2,000-seed rerun gave
   0.940.
3. We then added the confirmation stage. The final run flagged the same cell at stage 1, and
   it passed at stage 2 (0.939 direct, 0.944 peer).

**Follow-up (3,000 fresh seeds, oracle nuisances, partialling-out).** ER graphs cover at the
nominal rate, both with about 450 focal nodes (0.950 / 0.954) and about 1,800 (0.950 / 0.947).
The Barabási–Albert graph with about 310 focal nodes covers at 0.941 / 0.944, about two Monte
Carlo SEs below 0.95. Standard small-sample corrections barely change this: HC1 adds ≤0.1
points, t critical values ≤0.1, and HC3 ≤0.2. So the gap is *not* explained by the
small-sample bias of HC0. It appears specific to hub-heavy graphs, and its cause is not yet
known. It is small (under one point) and is tracked as an open question.

### Required evidence per release

| Benchmark | Criterion |
|---|---|
| `benchmarks/coverage.py` (oracle and correctly specified nuisances; ER, BA, SBM; continuous and binary T) | coverage criterion above |
| `benchmarks/design_choices.py` (default ML nuisances on four datasets) | default configuration: \|relative bias\| ≤ 5% and coverage ≥ 0.90 at 100 seeds (criterion set after the first run, which also chose the defaults; future releases must meet it unchanged) |
| `pytest -m slow` | the ER coverage cells, as a CI-sized version of the gate |

### Scenarios built to break the method

`make_homophily_trap` violates no-unobserved-confounding on purpose. The estimator *should*
be wrong there, and the criterion is that `negative_control_test` flags it. Planned
additions: two-hop interference, a misspecified exposure map, and dense graphs where the
focal set is tiny.

## Layer 4: realism (in progress)

* Story datasets with realistic structure: spatial, preferential attachment, communities.
* Planned: semi-synthetic Cora/Pubmed (real topology, simulated outcomes), and
  a benchmark against a randomized network experiment in which an observational subsample is
  analysed and compared with the experimental estimate.

## Layer 5: user-facing checks

`graphdml.selftest()` lets anyone re-run a signal check on their own machine: planted
effects versus a null twin, against fixed criteria (see the README).

`summary()` prints diagnostics and the identifying assumptions with every fit, and warns on
small focal sets, weak overlap, collinear residuals and isolated focal nodes. The
falsification checks in `graphdml.diagnostics` are tested to pass on clean data and to flag
the violations they target.

## Running everything

```bash
pip install -e ".[dev]"
pytest                       # layer 2 (~30 s)
pytest -m slow               # CI-sized coverage gate (~1 min)
cd benchmarks
python coverage.py           # full coverage gate (minutes, parallel)
python design_choices.py     # default-configuration evidence
```
