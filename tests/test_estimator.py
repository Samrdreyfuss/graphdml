"""Does the code compute the estimator in docs/methodology.md?

Reductions to independent implementations, oracle-nuisance checks and metamorphic tests.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from graphdml import GraphData, GraphDML, GraphDMLWarning
from graphdml.focal import dependency_matrix, select_focal_set
from graphdml.inference import ols_hc0
from graphdml.simulate import erdos_renyi, make_linear_gaussian, oracle_learners

from .helpers import linear_learners


# ---------------------------------------------------------------- final stage & variance
def test_sandwich_matches_statsmodels_hc0():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.RandomState(0)
    D = rng.standard_normal((500, 2))
    y = D @ np.array([1.0, -2.0]) + rng.standard_normal(500) * (1 + np.abs(D[:, 0]))
    coef, cov = ols_hc0(D, y)
    ref = sm.OLS(y, D).fit(cov_type="HC0")
    np.testing.assert_allclose(coef, ref.params)
    np.testing.assert_allclose(cov, ref.cov_params())


def test_partialling_out_cov_is_hc0_of_final_regression(linear_ds):
    sm = pytest.importorskip("statsmodels.api")
    m = GraphDML(**linear_learners(), exposure="mean", score="partialling-out",
                 random_state=0).fit(linear_ds.data)
    R = m.residuals_[["res_t", "res_peer:mean"]].to_numpy()
    ref = sm.OLS(m.residuals_["res_y"].to_numpy(), R).fit(cov_type="HC0")
    np.testing.assert_allclose(m.coef_, ref.params, rtol=1e-10)
    np.testing.assert_allclose(m.cov_, ref.cov_params(), rtol=1e-10)


# ---------------------------------------------------------------- reduction to standard DML
def _no_network_data(n=800, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, 3))
    T = X @ [0.5, -0.3, 0.2] + rng.standard_normal(n)
    Y = 1.5 * T + X @ [1.0, 0.5, -1.0] + rng.standard_normal(n)
    return GraphData(X, T, Y, sp.csr_array((n, n)))


def test_empty_graph_reduces_to_manual_dml():
    data = _no_network_data()
    m = GraphDML(LinearRegression(), LinearRegression(), featurizer="own", exposure=None,
                 score="partialling-out", random_state=0).fit(data)
    assert m.n_focal_ == data.n_nodes  # no edges -> every node is focal
    fold = np.empty(data.n_nodes, int)
    fold[m.focal_nodes_] = m.fold_assignment_
    res_t, res_y = np.empty(data.n_nodes), np.empty(data.n_nodes)
    for k in range(5):
        tr, te = fold != k, fold == k
        res_t[te] = data.T[te] - LinearRegression().fit(data.X[tr], data.T[tr]).predict(data.X[te])
        res_y[te] = data.Y[te] - LinearRegression().fit(data.X[tr], data.Y[tr]).predict(data.X[te])
    coef = (res_t @ res_y) / (res_t @ res_t)
    np.testing.assert_allclose(m.coef_[0], coef, rtol=1e-10)


def test_empty_graph_matches_doubleml():
    dml = pytest.importorskip("doubleml")
    data = _no_network_data()
    for score in ["partialling-out", "iv-type"]:
        m = GraphDML(LinearRegression(), LinearRegression(), featurizer="own", exposure=None,
                     score=score, random_state=0).fit(data)
        fold = np.empty(data.n_nodes, int)
        fold[m.focal_nodes_] = m.fold_assignment_
        smpls = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]
        obj = dml.DoubleMLData.from_arrays(data.X, data.Y, data.T)
        kwargs = {"ml_l": LinearRegression(), "ml_m": LinearRegression()}
        if score == "iv-type":
            kwargs["ml_g"] = LinearRegression()
        ref = dml.DoubleMLPLR(obj, n_folds=5, score="partialling out" if score ==
                              "partialling-out" else "IV-type", **kwargs)
        ref.set_sample_splitting([smpls])
        ref.fit()
        np.testing.assert_allclose(m.coef_[0], ref.coef[0], rtol=1e-8)
        np.testing.assert_allclose(m.se_[0], ref.se[0], rtol=1e-8)


# ---------------------------------------------------------------- recovery
def test_recovers_truth_with_correct_linear_nuisances(linear_ds):
    m = GraphDML(**linear_learners(), exposure="mean", random_state=0).fit(linear_ds.data)
    truth = np.array([1.0, 0.5])
    assert np.all(np.abs(m.coef_ - truth) < 4 * m.se_)
    assert m.coef_names_ == ["direct", "peer:mean"]
    assert m.ade_ == m.coef_[0] and m.ape_ == m.coef_[1]


@pytest.mark.parametrize("score", ["partialling-out", "iv-type"])
@pytest.mark.parametrize("aggregation", ["dml1", "dml2"])
@pytest.mark.parametrize("training", ["buffered", "focal"])
def test_all_settings_run_and_recover(linear_ds, score, aggregation, training):
    m = GraphDML(**linear_learners(), exposure="mean", score=score, aggregation=aggregation,
                 nuisance_training=training, random_state=0).fit(linear_ds.data)
    assert np.all(np.abs(m.coef_ - [1.0, 0.5]) < 5 * m.se_)


def test_oracle_nuisances_recover_truth(linear_ds):
    m = GraphDML(**oracle_learners(linear_ds), exposure="mean", random_state=0)
    m.fit(linear_ds.data)
    assert np.all(np.abs(m.coef_ - [1.0, 0.5]) < 4 * m.se_)


def test_binary_treatment_with_classifier():
    A = erdos_renyi(2000, 4.0, random_state=0)
    ds = make_linear_gaussian(A, treatment="binary", random_state=0)
    m = GraphDML(model_t=HistGradientBoostingClassifier(max_iter=50), exposure="mean",
                 random_state=0).fit(ds.data)
    assert 0 < m.diagnostics_["propensity_min"] <= m.diagnostics_["propensity_max"] < 1
    assert "treatment_auc" in m.diagnostics_


# ---------------------------------------------------------------- metamorphic
def _fit(data, **kw):
    return GraphDML(**{**linear_learners(), **kw}, exposure="mean", random_state=0).fit(data)


def test_scaling_outcome_scales_effects(linear_ds):
    d = linear_ds.data
    base, scaled = _fit(d), _fit(d.with_values(Y=3.0 * d.Y))
    np.testing.assert_allclose(scaled.coef_, 3.0 * base.coef_, rtol=1e-8)
    np.testing.assert_allclose(scaled.se_, 3.0 * base.se_, rtol=1e-8)


def test_shifting_treatment_leaves_effects_unchanged(linear_ds):
    # Exact for partialling-out (the treatment model's intercept absorbs the shift). For
    # IV-type, g must absorb a shift that differs for isolated nodes, which these linear
    # features cannot represent exactly.
    d = linear_ds.data
    po = {"score": "partialling-out"}
    base, shifted = _fit(d, **po), _fit(d.with_values(T=d.T + 5.0), **po)
    np.testing.assert_allclose(shifted.coef_, base.coef_, rtol=1e-8)


def test_adding_function_of_covariates_to_outcome_is_absorbed(linear_ds):
    d = linear_ds.data
    base, moved = _fit(d), _fit(d.with_values(Y=d.Y + d.X @ np.array([2.0, -1.0, 0.5])))
    np.testing.assert_allclose(moved.coef_, base.coef_, rtol=1e-6)


def test_two_disjoint_copies_halve_the_variance(linear_ds):
    d = linear_ds.data
    n = d.n_nodes
    dep = dependency_matrix(n, [d.pattern])
    focal = select_focal_set(dep, random_state=0)
    kw = oracle_learners(linear_ds)
    one = GraphDML(**kw, exposure="mean", focal_set=focal, random_state=0).fit(d)

    feats = np.vstack([kw["featurizer"].features] * 2)
    double = GraphData(
        np.vstack([d.X, d.X]), np.r_[d.T, d.T], np.r_[d.Y, d.Y],
        sp.block_diag([d.adjacency, d.adjacency], format="csr"),
    )
    kw2 = {**kw, "featurizer": type(kw["featurizer"])(feats)}
    two = GraphDML(**kw2, exposure="mean", focal_set=np.r_[focal, focal + n],
                   random_state=0).fit(double)
    np.testing.assert_allclose(two.coef_, one.coef_, rtol=1e-10)
    np.testing.assert_allclose(two.se_, one.se_ / np.sqrt(2), rtol=1e-10)


def test_relabeling_nodes_does_not_change_oracle_estimates(linear_ds):
    d = linear_ds.data
    perm = np.random.RandomState(0).permutation(d.n_nodes)
    inv = np.argsort(perm)
    dep = dependency_matrix(d.n_nodes, [d.pattern])
    focal = select_focal_set(dep, random_state=0)
    kw = oracle_learners(linear_ds)
    base = GraphDML(**kw, exposure="mean", focal_set=focal, random_state=0).fit(d)
    A = d.adjacency[perm][:, perm]
    permuted = GraphData(d.X[perm], d.T[perm], d.Y[perm], A)
    kw2 = {**kw, "featurizer": type(kw["featurizer"])(kw["featurizer"].features[perm])}
    other = GraphDML(**kw2, exposure="mean", focal_set=inv[focal], random_state=0).fit(permuted)
    np.testing.assert_allclose(other.coef_, base.coef_, rtol=1e-10)


# ---------------------------------------------------------------- API behaviour
def test_invalid_explicit_focal_set_raises(linear_ds):
    i, j = linear_ds.data.adjacency.nonzero()
    with pytest.raises(ValueError, match="overlapping"):
        GraphDML(**linear_learners(), focal_set=[i[0], j[0]]).fit(linear_ds.data)


def test_classifier_with_continuous_treatment_raises(linear_ds):
    with pytest.raises(ValueError, match="classifier"):
        GraphDML(model_t=HistGradientBoostingClassifier()).fit(linear_ds.data)


def test_iid_mode_warns(linear_ds):
    with pytest.warns(GraphDMLWarning, match="independent"):
        GraphDML(**linear_learners(), focal_set=None, random_state=0).fit(linear_ds.data)


def test_original_mode_settings(linear_ds):
    m = GraphDML(LinearRegression(), LinearRegression(), mode="original", random_state=0)
    m.fit(linear_ds.data)
    s = m.settings_
    assert (s["focal_set"], s["nuisance_training"], s["aggregation"], s["n_folds"]) == (
        "random", "focal", "dml1", 3,
    )
    assert m.get_params()["n_folds"] == 5  # user-facing params untouched


def test_repetitions_and_summary(linear_ds):
    m = GraphDML(**linear_learners(), exposure="mean", n_rep=3, random_state=0)
    m.fit(linear_ds.data)
    assert m.coef_reps_.shape == (3, 2)
    assert np.allclose(m.coef_, np.median(m.coef_reps_, axis=0))
    assert np.all(m.se_ >= np.median(m.se_reps_, axis=0) - 1e-12)
    text = str(m.summary())
    assert "direct" in text and "Identifying assumptions" in text
    assert set(m.residuals_.columns) >= {"node_id", "fold", "res_t", "res_peer:mean", "res_y"}


def test_not_fitted():
    with pytest.raises(NotFittedError):
        GraphDML().summary()
    assert not hasattr(GraphDML(), "ade_")


def test_estimation_nodes_restrict_focal_set(linear_ds):
    mask = np.zeros(linear_ds.data.n_nodes, bool)
    mask[:1500] = True
    m = GraphDML(**linear_learners(), exposure="mean", estimation_nodes=mask, random_state=0)
    m.fit(linear_ds.data)
    assert m.focal_nodes_.max() < 1500
    assert np.all(np.abs(m.coef_ - [1.0, 0.5]) < 4 * m.se_)


def test_cluster_se_matches_statsmodels_and_singletons_match_hc0(linear_ds):
    sm = pytest.importorskip("statsmodels.api")
    m = GraphDML(**linear_learners(), exposure="mean", score="partialling-out",
                 random_state=0).fit(linear_ds.data)
    singletons = np.arange(linear_ds.data.n_nodes)
    np.testing.assert_allclose(m.cluster_summary_frame(singletons)["std_err"], m.se_)
    groups = np.arange(linear_ds.data.n_nodes) // 50
    R = m.residuals_[["res_t", "res_peer:mean"]].to_numpy()
    ref = sm.OLS(m.residuals_["res_y"].to_numpy(), R).fit(
        cov_type="cluster", cov_kwds={"groups": groups[m.focal_nodes_], "use_correction": False})
    np.testing.assert_allclose(m.cluster_summary_frame(groups)["std_err"], ref.bse, rtol=1e-8)
