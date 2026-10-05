"""Default nuisance learners.

Any scikit-learn regressor or classifier works as a GraphDML nuisance model. The defaults
are histogram gradient boosting with a lower learning rate, larger leaves and early
stopping. Over-fitted nuisances, especially over-confident propensities, add noise to the
treatment residual, which biases the partialling-out estimates toward zero; the extra
regularisation measurably reduces that bias (see ``docs/validation-results.md``).
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

__all__ = ["default_classifier", "default_regressor"]

_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "min_samples_leaf": 50,
    "early_stopping": True,
}


def default_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**_HGB_PARAMS)


def default_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**_HGB_PARAMS)
