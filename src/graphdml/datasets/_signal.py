"""A synthetic dataset with planted effects, and a null twin with none.

Used by :func:`graphdml.selftest` to check, on your machine, that the estimator finds
real signals and does not invent false ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.utils import check_random_state

from graphdml.data import GraphData
from graphdml.datasets._base import GraphDataset, load_descr
from graphdml.exposure import row_normalize
from graphdml.simulate.graphs import erdos_renyi

__all__ = ["make_signal_check"]


def make_signal_check(
    n: int = 6000,
    *,
    direct_effect: float = 1.0,
    peer_effect: float = 0.6,
    confounding: float = 1.0,
    avg_degree: float = 4.0,
    noise_sd: float = 2.0,
    random_state: Any = 0,
) -> GraphDataset:
    """Planted direct and peer effects, hidden behind strong network confounding.

    Set ``direct_effect=0, peer_effect=0`` for the **null twin**: the same data-generating
    process with no true effect. Methods that ignore the network still "find" effects in
    the null twin, because the confounding is strong; a correct method should not.

    ``T`` is binary. Both treatment and outcome depend nonlinearly on a node's own
    covariates and on its neighbors' (mean and maximum), with strength ``confounding``.
    The exposure is ``"mean"``: the share of treated neighbors.
    """
    rng = check_random_state(random_state)
    A = erdos_renyi(n, avg_degree, rng)
    M = row_normalize(A)
    X = rng.standard_normal((n, 3))
    nbr_mean = M @ X
    nbr_max = np.zeros_like(X)
    has = np.diff(A.indptr) > 0
    nbr_max[has] = np.maximum.reduceat(X[A.indices], A.indptr[:-1][has], axis=0)

    logit = (
        -0.2
        + 0.5 * X[:, 0]
        + 0.4 * X[:, 1] * X[:, 2]
        + confounding * (0.9 * nbr_mean[:, 0] + 0.5 * nbr_mean[:, 1])
    )
    T = rng.binomial(1, expit(logit)).astype(float)
    g = (
        1.0 * X[:, 0]
        + 0.8 * np.sin(2 * X[:, 1])
        + 0.5 * X[:, 2] ** 2
        + confounding * (1.6 * nbr_mean[:, 0] + 1.0 * np.tanh(nbr_max[:, 1]))
    )
    Y = g + direct_effect * T + peer_effect * (M @ T) + noise_sd * rng.standard_normal(n)
    return GraphDataset(
        name="signal_check" if (direct_effect or peer_effect) else "signal_check_null",
        data=GraphData(X, T, Y, A, feature_names=["x1", "x2", "x3"]),
        truth={"ade": direct_effect, "ape": peer_effect},
        exposure="mean",
        treatment_name="treated",
        outcome_name="outcome",
        DESCR=load_descr("signal_check"),
    )
