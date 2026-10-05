import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from graphdml import NeighborhoodFeatures
from graphdml.simulate import erdos_renyi, make_linear_gaussian


def linear_learners() -> dict:
    """Correctly specified nuisances for ``make_linear_gaussian`` (continuous T, mean)."""
    return {
        "featurizer": NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
        "model_y": LinearRegression(),
        "model_t": LinearRegression(),
    }


@pytest.fixture
def linear_ds():
    A = erdos_renyi(3000, 4.0, random_state=1)
    return make_linear_gaussian(A, theta=1.0, alpha=0.5, random_state=2)


@pytest.fixture
def rng():
    return np.random.RandomState(0)
