"""Graph generators, data-generating processes with known truth, and oracle nuisances."""

from graphdml.simulate.dgp import (
    make_benchmark_linear,
    make_benchmark_nonlinear,
    make_linear_gaussian,
)
from graphdml.simulate.graphs import (
    barabasi_albert,
    edges_to_adjacency,
    erdos_renyi,
    geometric_graph,
    random_geometric,
    stochastic_block_model,
)
from graphdml.simulate.oracle import OracleRegressor, oracle_learners

__all__ = [
    "OracleRegressor",
    "barabasi_albert",
    "edges_to_adjacency",
    "erdos_renyi",
    "geometric_graph",
    "make_benchmark_linear",
    "make_benchmark_nonlinear",
    "make_linear_gaussian",
    "oracle_learners",
    "random_geometric",
    "stochastic_block_model",
]
