import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import graphdml
from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.datasets import make_signal_check


def test_signal_check_planted_and_null_twin():
    planted = make_signal_check(2000, random_state=3)
    null = make_signal_check(2000, direct_effect=0.0, peer_effect=0.0, random_state=3)
    assert planted.truth == {"ade": 1.0, "ape": 0.6}
    assert null.truth == {"ade": 0.0, "ape": 0.0}
    assert planted.name == "signal_check" and null.name == "signal_check_null"
    # same graph and covariates; only the outcome differs
    np.testing.assert_array_equal(planted.data.X, null.data.X)
    np.testing.assert_array_equal(planted.data.T, null.data.T)
    assert (planted.data.adjacency != null.data.adjacency).nnz == 0
    assert not np.allclose(planted.data.Y, null.data.Y)
    assert planted.DESCR.startswith("# Signal Check")
    again = make_signal_check(2000, random_state=3)
    np.testing.assert_array_equal(planted.data.Y, again.data.Y)


def test_confounding_fools_a_network_blind_estimate():
    null = make_signal_check(6000, direct_effect=0.0, peer_effect=0.0, random_state=1)
    naive = graphdml.baselines.naive_ols(null.data, "mean")
    z = naive["coef"] / naive["std_err"]
    assert (np.abs(z) > 4).all()  # large "effects" where none exist


def test_selftest_runs_and_reports():
    est = GraphDML(
        LinearRegression(), LinearRegression(),
        featurizer=NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
        exposure="mean",
    )
    result = graphdml.selftest(n_reps=3, n=2500, estimator=est, n_jobs=1)
    assert list(result.table.index.names) == ["world", "effect"]
    assert set(result.table.index.get_level_values("world")) == {"planted", "null"}
    assert len(result.checks) == 8 and result.checks["ok"].dtype == bool
    assert "graphdml self-test" in repr(result)
    # the network-blind baseline is fooled in the null world
    assert result.table.loc[("null", "direct"), "naive_detected"] == 1.0


@pytest.mark.slow
def test_selftest_passes_with_default_settings():
    result = graphdml.selftest(n_reps=30)
    assert result.passed, result
