"""Graph generators, data-generating processes with known truth, and oracle nuisances."""

from graphdml.simulate.dgp import make_linear_gaussian, make_paper_linear, make_paper_nonlinear
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
    "make_linear_gaussian",
    "make_paper_linear",
    "make_paper_nonlinear",
    "oracle_learners",
    "random_geometric",
    "stochastic_block_model",
]
