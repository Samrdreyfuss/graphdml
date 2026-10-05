"""Pre-registered validation gates (see docs/validation.md). Run with ``pytest -m slow``.

Seeds are fixed, so results are deterministic for a given library version; a gate that
starts failing after a change means the change altered the estimator's behaviour.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.simulate import erdos_renyi, make_linear_gaussian, oracle_learners

pytestmark = pytest.mark.slow

SEEDS = range(300)
# 0.95 +/- 3 Monte Carlo standard errors (docs/validation.md, "Coverage criterion").
_MC_SE = np.sqrt(0.95 * 0.05 / len(SEEDS))
BAND = (0.95 - 3 * _MC_SE, 0.95 + 3 * _MC_SE)


def _coverage(make_model):
    covered = []
    for seed in SEEDS:
        A = erdos_renyi(2000, 4.0, random_state=seed)
        ds = make_linear_gaussian(A, theta=1.0, alpha=0.5, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_model(ds, seed).fit(ds.data)
        ci = m.conf_int().to_numpy()
        covered.append((ci[:, 0] <= [1.0, 0.5]) & ([1.0, 0.5] <= ci[:, 1]))
    return np.mean(covered, axis=0)


def test_gate_oracle_nuisances_cover_nominally():
    cov = _coverage(lambda ds, s: GraphDML(**oracle_learners(ds), exposure="mean",
                                           random_state=s))
    assert np.all((BAND[0] <= cov) & (cov <= BAND[1])), cov


def test_gate_correct_linear_nuisances_cover_nominally():
    def make(ds, s):
        return GraphDML(
            LinearRegression(), LinearRegression(),
            featurizer=NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
            exposure="mean", random_state=s,
        )

    cov = _coverage(make)
    assert np.all((BAND[0] <= cov) & (cov <= BAND[1])), cov
