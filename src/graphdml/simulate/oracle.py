"""Oracle nuisance models: plug the *true* nuisance functions into GraphDML.

Running GraphDML with oracle nuisances isolates the final stage and the variance formula
from machine-learning error: with oracles, intervals must cover at the nominal rate. Any
nuisance can instead be replaced by a constant (deliberately misspecified) model to test
the robustness properties of each score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

from graphdml.datasets._base import GraphDataset
from graphdml.features import PrecomputedFeatures

__all__ = ["OracleRegressor", "oracle_learners"]


class OracleRegressor(RegressorMixin, BaseEstimator):
    """Ignores training data and predicts one column of the feature matrix."""

    def __init__(self, column: int = 0) -> None:
        self.column = column

    def fit(self, X: Any, y: Any) -> OracleRegressor:
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, self.column]


def oracle_learners(
    dataset: GraphDataset,
    *,
    treatment: bool = True,
    outcome: bool = True,
    g: bool = True,
) -> dict[str, Any]:
    """GraphDML keyword arguments that use the true nuisances from ``dataset.extras``.

    Setting ``treatment``, ``outcome`` or ``g`` to False replaces that nuisance with a
    constant (training-mean) model, i.e. a badly misspecified one.
    """
    ex = dataset.extras
    features = np.column_stack([ex["m"], ex["ell"], ex["g"]])
    return {
        "featurizer": PrecomputedFeatures(features, names=["m", "ell", "g"]),
        "model_t": OracleRegressor(0) if treatment else DummyRegressor(),
        "model_y": OracleRegressor(1) if outcome else DummyRegressor(),
        "model_g": OracleRegressor(2) if g else DummyRegressor(),
    }
