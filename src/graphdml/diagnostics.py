"""Falsification checks for a GraphDML analysis.

None of these can *prove* the identifying assumptions. Each one targets a specific way an
analysis can go wrong and flags it when the data contradict the assumptions:

* :func:`placebo_test` - shuffled treatments must show no effect (pipeline sanity).
* :func:`negative_control_test` - an outcome the treatment cannot affect (e.g. measured
  before treatment) must show no effect; a non-zero "effect" signals confounding, such as
  latent homophily.
* :func:`two_hop_test` - adds the treatments of two-hop neighbors as an extra exposure;
  a non-zero coefficient contradicts the one-hop interference assumption (or signals
  confounding that is correlated along the network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.utils import check_random_state

from graphdml.data import GraphData
from graphdml.exposure import MeanExposure, TwoHopExposure, WeightedExposure, resolve_exposures

__all__ = ["FalsificationResult", "negative_control_test", "placebo_test", "two_hop_test"]


@dataclass
class FalsificationResult:
    """Outcome of a falsification check. ``passed=False`` means the check raised a flag."""

    name: str
    passed: bool
    message: str
    table: pd.DataFrame

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FLAG"
        return f"{self.name}: {verdict}. {self.message}\n{self.table.to_string()}"


def placebo_test(
    estimator: Any,
    data: GraphData,
    n_permutations: int = 5,
    alpha: float = 0.05,
    random_state: Any = 0,
) -> FalsificationResult:
    """Refit with randomly permuted treatments; every effect should be ~0.

    This checks the estimation pipeline (e.g. leakage, a broken featurizer), not the
    causal assumptions. Uses a Bonferroni correction across permutations and effects.
    """
    rng = check_random_state(random_state)
    rows = []
    for p in range(n_permutations):
        est = clone(estimator).fit(data.with_values(T=rng.permutation(data.T)))
        frame = est.summary_frame(alpha, total=False).assign(permutation=p)
        rows.append(frame.reset_index())
    table = pd.concat(rows, ignore_index=True)
    passed = _bonferroni_pass(table, alpha)
    msg = (
        "Effects of shuffled treatments are consistent with zero."
        if passed
        else "Shuffled treatments show non-zero effects: check for leakage in features or data."
    )
    return FalsificationResult("Placebo (permuted treatment)", passed, msg, table)


def negative_control_test(
    estimator: Any, data: GraphData, outcome: Any, alpha: float = 0.05
) -> FalsificationResult:
    """Refit with an outcome the treatment cannot have affected; every effect should be ~0.

    Good negative controls share the confounders of the real outcome but are causally
    unaffected by treatment, e.g. the same outcome measured before treatment.
    """
    est = clone(estimator).fit(data.with_values(Y=outcome))
    table = est.summary_frame(alpha, total=False).reset_index()
    passed = _bonferroni_pass(table, alpha)
    msg = (
        "No effect on the negative-control outcome."
        if passed
        else "Treatment 'affects' an outcome it cannot affect: evidence of unobserved "
        "confounding (e.g. homophily). Treat the main estimates with caution."
    )
    return FalsificationResult("Negative-control outcome", passed, msg, table)


def two_hop_test(estimator: Any, data: GraphData, alpha: float = 0.05) -> FalsificationResult:
    """Test whether two-hop neighbors' treatments matter once one-hop exposure is modelled.

    The refit uses a focal set that is independent at the two-hop scale (pairwise distance
    >= 5), so it has less power than the main analysis.
    """
    exposures = resolve_exposures(estimator.get_params()["exposure"])
    if not exposures:
        raise ValueError("two_hop_test needs an estimator with a peer exposure.")
    first = exposures[0]
    averaged = isinstance(first, MeanExposure) or (
        isinstance(first, WeightedExposure) and first.normalize
    )
    two_hop = TwoHopExposure("mean" if averaged else "sum")
    est = clone(estimator).set_params(exposure=[*exposures, two_hop]).fit(data)
    table = est.summary_frame(alpha, total=False).reset_index()
    table["n_focal"] = est.n_focal_
    p = float(table.iloc[-1]["p_value"])
    passed = p >= alpha
    msg = (
        f"No evidence of two-hop effects (p = {p:.3f})."
        if passed
        else f"Two-hop neighbors' treatments predict the outcome (p = {p:.3g}): interference "
        "may extend beyond one hop, or confounding is correlated along the network."
    )
    return FalsificationResult("Two-hop interference", passed, msg, table)


def _bonferroni_pass(table: pd.DataFrame, alpha: float) -> bool:
    z_crit = stats.norm.ppf(1 - alpha / (2 * len(table)))
    return bool((np.abs(table["z"]) < z_crit).all())
