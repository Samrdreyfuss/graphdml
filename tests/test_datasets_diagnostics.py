import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from graphdml import (
    GraphDML,
    NeighborhoodFeatures,
    negative_control_test,
    placebo_test,
    two_hop_test,
)
from graphdml.baselines import compare_methods, naive_ols
from graphdml.datasets import (
    GraphDataset,
    make_classroom_tutoring,
    make_flu_town,
    make_homophily_trap,
    make_referral_app,
    make_toy_graph,
)
from graphdml.exposure import TwoHopExposure
from graphdml.focal import dependency_matrix, is_independent, select_focal_set
from graphdml.simulate import erdos_renyi, make_linear_gaussian

from .helpers import linear_learners

MAKERS = [make_flu_town, make_referral_app, make_classroom_tutoring, make_homophily_trap]


@pytest.mark.parametrize("maker", [*MAKERS, make_toy_graph])
def test_datasets_are_reproducible_and_documented(maker):
    a, b = maker(random_state=3), maker(random_state=3)
    assert isinstance(a, GraphDataset)
    np.testing.assert_array_equal(a.data.Y, b.data.Y)
    assert set(a.truth) == {"ade", "ape"}
    assert a.DESCR.strip(), f"{a.name} is missing its story card"
    assert {a.treatment_name, a.outcome_name} <= set(a.nodes_df.columns)
    assert repr(a).startswith("GraphDataset(")


@pytest.mark.parametrize("maker", MAKERS[:3])
def test_story_datasets_are_recovered_and_naive_ols_is_not(maker):
    ds = maker(random_state=0)
    m = GraphDML(exposure=ds.exposure, random_state=0).fit(ds.data)
    truth = np.array([ds.truth["ade"], ds.truth["ape"]])
    assert np.all(np.abs(m.coef_ - truth) < 3 * m.se_)
    naive = naive_ols(ds.data, ds.exposure)
    misses = (naive["ci_lower"] > truth) | (naive["ci_upper"] < truth)
    assert misses.any()


def test_heterogeneous_classroom_focal_truth_differs():
    ds = make_classroom_tutoring(heterogeneous=True, random_state=0)
    m = GraphDML(exposure="mean", random_state=0).fit(ds.data)
    assert ds.focal_truth(m.focal_nodes_)["ade"] > ds.truth["ade"] + 0.2


def test_toy_graph_focal_set():
    ds = make_toy_graph()
    dep = dependency_matrix(30, [ds.data.pattern])
    focal = select_focal_set(dep, random_state=0)
    assert is_independent(dep, focal) and 28 in focal and 29 in focal


def test_negative_control_flags_homophily():
    ds = make_homophily_trap(3000, random_state=0)
    m = GraphDML(exposure="mean", random_state=0).fit(ds.data)
    assert m.ape_ > 1.0  # spurious: the true peer effect is zero
    result = negative_control_test(GraphDML(exposure="mean", random_state=0), ds.data,
                                   ds.extras["negative_control"])
    assert not result.passed
    assert "FLAG" in repr(result)


def test_placebo_passes_on_clean_data(linear_ds):
    result = placebo_test(GraphDML(**linear_learners(), exposure="mean"), linear_ds.data,
                          n_permutations=3)
    assert result.passed
    assert len(result.table) == 6


def test_two_hop_test_passes_without_and_flags_with_two_hop_effects():
    A = erdos_renyi(6000, 3.0, random_state=0)
    ds = make_linear_gaussian(A, random_state=0)
    kw = {
        **linear_learners(),
        "featurizer": NeighborhoodFeatures(aggs=("mean",), hops=3, include_degree=False),
        "exposure": "mean",
        "random_state": 0,
    }
    assert two_hop_test(GraphDML(**kw), ds.data).passed
    two_hop = TwoHopExposure("mean")(ds.data)
    flagged = two_hop_test(GraphDML(**kw), ds.data.with_values(Y=ds.data.Y + 2.0 * two_hop))
    assert not flagged.passed
    assert flagged.table["effect"].iloc[-1] == "peer:two_hop_mean"


def test_compare_methods_table():
    ds = make_flu_town(2000, random_state=0)
    table = compare_methods(ds.data, exposure="sum", truth=ds.truth,
                            estimator=GraphDML(LinearRegression(), exposure="sum",
                                               random_state=0))
    assert set(table["method"]) >= {"Naive OLS (no network)", "GraphDML"}
    assert {"truth", "covers_truth"} <= set(table.columns)


def test_lastfm_promo_if_cached():
    from graphdml.datasets._fetch import get_data_home

    if not (get_data_home() / "lastfm_asia.zip").exists():
        pytest.skip("LastFM Asia not downloaded (run make_lastfm_promo() once)")
    from graphdml.datasets import make_lastfm_promo

    ds = make_lastfm_promo(random_state=0)
    assert ds.data.n_nodes == 7624 and ds.data.n_edges == 27806
    assert ds.DESCR.strip()
    m = GraphDML(exposure="mean", random_state=0).fit(ds.data)
    assert np.all(np.abs(m.coef_ - [2.0, 3.0]) < 3 * m.se_)


def test_insurance_experiment_if_available():
    import os
    from pathlib import Path

    path = Path(os.environ.get("GRAPHDML_INSURANCE_DATA",
                               "~/Downloads/113593-V1/data/data")).expanduser()
    if not (path / "0422survey.dta").exists():
        pytest.skip("insurance replication data not available")
    from graphdml.datasets import load_insurance_experiment

    ds = load_insurance_experiment(path)
    d, ex = ds.data, ds.extras
    assert d.n_nodes == 4902 and d.directed
    pr = ex["published_exposure"]
    ok = ~np.isnan(pr)
    assert np.mean(np.isclose(ex["exposure"](d)[ok], pr[ok])) > 0.95
    m = GraphDML(exposure=None, estimation_nodes=~ex["second_round"], random_state=0).fit(d)
    assert abs(m.ade_ - ex["published"]["ade_first_round"]["coef"]) < 2 * m.se_[0]
