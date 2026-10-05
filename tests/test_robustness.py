"""Question 1b: which nuisance can be wrong without biasing each score?

Derivation (docs/methodology.md, section 4): the partialling-out score stays consistent
when the treatment model is right and the outcome model is wrong, but is attenuated by
E[V^2] / (E[V^2] + E[delta^2]) when the treatment model is off by delta. The IV-type score
stays consistent when either the treatment model or g is right.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from graphdml import GraphData, GraphDML
from graphdml.datasets import GraphDataset
from graphdml.simulate import erdos_renyi, make_linear_gaussian, oracle_learners


@pytest.fixture(scope="module")
def ds():
    A = erdos_renyi(8000, 4.0, random_state=0)
    return make_linear_gaussian(A, theta=1.0, alpha=0.5, random_state=0)


def _fit(ds, score, **which):
    m = GraphDML(**oracle_learners(ds, **which), exposure="mean", score=score, random_state=0)
    return m.fit(ds.data)


def _z(m):
    return np.abs(m.coef_ - [1.0, 0.5]) / m.se_


def test_partialling_out_survives_wrong_outcome_model(ds):
    assert np.all(_z(_fit(ds, "partialling-out", outcome=False)) < 4)


def test_partialling_out_is_attenuated_by_wrong_treatment_model(ds):
    m = _fit(ds, "partialling-out", treatment=False)
    assert _z(m)[0] > 10
    assert m.coef_[0] < 0.6  # truth 1.0


def test_attenuation_factor_matches_derivation_without_network():
    # Single regressor: theta_hat -> theta * E[V^2] / (E[V^2] + E[delta^2]).
    rng = np.random.RandomState(0)
    n = 40_000
    X = rng.standard_normal((n, 3))
    m_true = X @ np.array([0.6, -0.6, 0.6])
    T = m_true + rng.standard_normal(n)
    g = X @ np.array([1.0, 0.5, -1.0])
    Y = 1.0 * T + g + rng.standard_normal(n)
    data = GraphData(X, T, Y, sp.csr_array((n, n)))
    ds = GraphDataset("no_network", data, {"ade": 1.0}, "none",
                      extras={"m": m_true, "ell": m_true + g, "g": g})
    m = GraphDML(**oracle_learners(ds, treatment=False), exposure=None,
                 score="partialling-out", random_state=0)
    m.fit(data)
    expected = 1.0 / (1.0 + np.var(m_true))
    np.testing.assert_allclose(m.coef_[0], expected, rtol=0.03)


def test_iv_type_survives_wrong_treatment_model(ds):
    assert np.all(_z(_fit(ds, "iv-type", treatment=False)) < 4)


def test_iv_type_survives_wrong_g(ds):
    assert np.all(_z(_fit(ds, "iv-type", g=False)) < 4)


def test_both_wrong_is_biased(ds):
    assert _z(_fit(ds, "iv-type", treatment=False, g=False))[0] > 10
