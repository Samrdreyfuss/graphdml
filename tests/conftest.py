import pytest

from graphdml.simulate import erdos_renyi, make_linear_gaussian


@pytest.fixture
def linear_ds():
    A = erdos_renyi(3000, 4.0, random_state=1)
    return make_linear_gaussian(A, theta=1.0, alpha=0.5, random_state=2)
