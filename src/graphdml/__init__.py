"""graphdml: double machine learning for causal effects on networks.

Estimate the average direct effect of a node's own treatment and the average peer effect
of its neighbors' treatments, with confidence intervals that account for network
dependence.
"""

from graphdml.data import GraphData
from graphdml.diagnostics import negative_control_test, placebo_test, two_hop_test
from graphdml.estimator import GraphDML
from graphdml.exceptions import GraphDMLWarning
from graphdml.exposure import (
    ExposureMap,
    MatrixExposure,
    MeanExposure,
    SumExposure,
    TwoHopExposure,
    WeightedExposure,
)
from graphdml.features import NeighborhoodFeatures, OwnFeatures, PrecomputedFeatures
from graphdml.selftest import selftest

__version__ = "0.2.0"

__all__ = [
    "ExposureMap",
    "GraphDML",
    "GraphDMLWarning",
    "GraphData",
    "MatrixExposure",
    "MeanExposure",
    "NeighborhoodFeatures",
    "OwnFeatures",
    "PrecomputedFeatures",
    "SumExposure",
    "TwoHopExposure",
    "WeightedExposure",
    "__version__",
    "negative_control_test",
    "placebo_test",
    "selftest",
    "two_hop_test",
]
