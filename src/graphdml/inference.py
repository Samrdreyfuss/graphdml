"""Final-stage estimation and variance for linear (in the target) moment conditions.

All GraphDML scores have the form ``psi_i(zeta) = R_i * (y_i - D_i' zeta)`` where ``R_i``
are residualised regressors ("instruments"), ``D_i`` the regressors and ``y_i`` the
(possibly residualised) outcome:

* partialling-out: ``R = D = [res_T, res_Z...]``, ``y = res_Y``
* IV-type:         ``R = [res_T, res_Z...]``, ``D = [T, Z...]``, ``y = Y - g_hat``

The sandwich covariance ``J^{-1} Omega J^{-T} / n`` with ``J = R'D / n`` and
``Omega = sum_i u_i^2 R_i R_i' / n`` is the standard DML sandwich. For
partialling-out it equals the HC0 covariance of the OLS of ``res_Y`` on the residuals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["aggregate_repetitions", "normal_ci", "ols_hc0", "sandwich_cov", "solve_moment"]


def solve_moment(R: np.ndarray, D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``sum_i R_i (y_i - D_i' zeta) = 0``."""
    return np.linalg.solve(R.T @ D, R.T @ y)


def sandwich_cov(
    R: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Covariance of ``coef`` from the empirical sandwich formula.

    With ``groups``, scores are summed within groups before forming ``Omega``
    (cluster-robust, CR0), allowing arbitrary dependence within each group.
    """
    n = R.shape[0]
    u = y - D @ coef
    J = R.T @ D / n
    Ru = R * u[:, None]
    if groups is not None:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        Ru = np.vstack([np.bincount(inv, weights=Ru[:, j]) for j in range(Ru.shape[1])]).T
    omega = Ru.T @ Ru / n
    J_inv = np.linalg.inv(J)
    return J_inv @ omega @ J_inv.T / n


def ols_hc0(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients with HC0 (heteroskedasticity-robust) covariance."""
    coef = solve_moment(D, D, y)
    return coef, sandwich_cov(D, D, y, coef)


def normal_ci(
    coef: np.ndarray, se: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    z = stats.norm.ppf(1 - alpha / 2)
    return coef - z * se, coef + z * se


def aggregate_repetitions(
    coefs: np.ndarray, covs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Median aggregation over repeated cross-fitting (Chernozhukov et al., 2018, Sec. 3.4).

    ``coef = median_s coef_s`` and ``cov = median_s (cov_s + d_s d_s')`` element-wise,
    with ``d_s = coef_s - coef``. The variance adjustment accounts for split-to-split
    variability.
    """
    coef = np.median(coefs, axis=0)
    d = coefs - coef
    adjusted = covs + d[:, :, None] * d[:, None, :]
    return coef, np.median(adjusted, axis=0)
