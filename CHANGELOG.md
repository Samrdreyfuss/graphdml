# Changelog

All notable changes to graphdml are listed here. Versions follow
[semantic versioning](https://semver.org/); before 1.0, minor versions may change behavior.

## [Unreleased]

## [0.2.0] - 2026-10-09

### Added
- Documentation site (https://samrdreyfuss.github.io/graphdml/): guides, methodology,
  validation, example datasets and an API reference generated from the docstrings.
- `graphdml.selftest()` and `make_signal_check()`: a built-in check, on synthetic data with
  planted effects and a null twin that has none, that the estimator finds real signals and
  does not invent false ones. Reports bias, coverage, power and false-alarm rates against
  fixed criteria, next to a network-blind baseline.
- **ATE (total effect)**: the average effect of treating every node vs none, reported first
  in `summary()` and as a `"total"` row in `summary_frame()`, `cluster_summary_frame()` and
  `conf_int()`, with `model.ate_` and `model.total_weights_`. Pass `total=False` for the
  direct and peer rows only.
- Practitioner-facing labels in `summary()`: `ATE (total)`, `direct (ADE)`, `peer (APE)`,
  with a plain-language unit line for each and a note on z-statistics.
- `GraphData.with_values(..., feature_names=...)`; covariates may change their number of
  columns.
- Installation guide, badges and project links in the README and on PyPI.
- README datasets table now says which datasets need outside data (LastFM downloads on
  first use, the insurance experiment must be downloaded from openICPSR), and lists
  `make_toy_graph`.

### Improved
- Repository polish: code of conduct, security policy, issue forms and a pull-request
  template; `py.typed` marker so type checkers use graphdml's annotations; CI pulls the Neo4j
  test image through a mirror, which fixes random failures from Docker Hub rate limits.
- `load_insurance_experiment` explains where to get the data when the files are missing,
  and `fetch_lastfm_asia` explains what to do when the download fails.

### Changed
- `summary_frame()` includes a `"total"` row by default when a peer exposure is fitted.
- Removed the unused `gnn` and `viz` optional extras.

### Fixed
- The placebo test in the test suite is seeded, so it no longer fails occasionally.

## [0.1.0] - 2026-10-06

First release.

- `GraphDML` estimator for direct and peer effects on networks: focal sets from dependency
  sets, cross-fitting with buffered nuisance training, IV-type and partialling-out scores,
  sandwich and cluster-robust standard errors, sub-population estimation, and
  `mode="original"` for the original GDML procedure.
- Nuisance models from any scikit-learn estimator, with neighborhood features.
- Diagnostics and falsification checks: `placebo_test`, `negative_control_test`,
  `two_hop_test`.
- Example datasets with known truth, a semi-synthetic LastFM Asia dataset, and a loader for
  the Cai, de Janvry & Sadoulet (2015) randomized network experiment.
- Neo4j integration: load graphs, write results back, GDS FastRP features, demo.
- Validation suite and benchmarks.

[Unreleased]: https://github.com/Samrdreyfuss/graphdml/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Samrdreyfuss/graphdml/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Samrdreyfuss/graphdml/releases/tag/v0.1.0
