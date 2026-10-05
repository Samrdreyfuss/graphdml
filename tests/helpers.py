"""Shared test helpers."""

from sklearn.linear_model import LinearRegression

from graphdml import NeighborhoodFeatures


def linear_learners() -> dict:
    """Correctly specified nuisances for ``make_linear_gaussian`` (continuous T, mean)."""
    return {
        "featurizer": NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
        "model_y": LinearRegression(),
        "model_t": LinearRegression(),
    }
