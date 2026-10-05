"""Semi-synthetic datasets: a real network and real covariates, simulated T and Y."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

from graphdml.data import GraphData
from graphdml.datasets._base import GraphDataset, load_descr
from graphdml.datasets._fetch import fetch_lastfm_asia
from graphdml.exposure import row_normalize
from graphdml.features import _aggregate

__all__ = ["lastfm_covariates", "make_lastfm_promo"]

_COVARIATE_CACHE: dict[str, tuple[np.ndarray, list[str]]] = {}


def lastfm_covariates(data_home: str | os.PathLike | None = None, n_taste: int = 6):
    """Real covariates for LastFM Asia: listening activity and taste dimensions.

    ``activity`` is the standardised log number of liked artists; ``taste_1..k`` are
    standardised truncated-SVD components of the (row-normalised) user × artist matrix.
    Returns ``(network, X, feature_names)``. Deterministic.
    """
    net = fetch_lastfm_asia(data_home)
    key = f"{data_home}:{n_taste}"
    if key not in _COVARIATE_CACHE:
        likes = net.attributes["likes"]
        activity = np.log1p(np.asarray(likes.sum(axis=1)).ravel())
        svd = TruncatedSVD(n_components=n_taste, random_state=0)
        taste = svd.fit_transform(normalize(likes))
        X = np.column_stack([activity, taste])
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        names = ["activity"] + [f"taste_{k + 1}" for k in range(n_taste)]
        _COVARIATE_CACHE[key] = (X, names)
    X, names = _COVARIATE_CACHE[key]
    return net, X.copy(), list(names)


def make_lastfm_promo(
    *,
    confounding: float = 1.0,
    direct_effect: float = 2.0,
    peer_effect: float = 3.0,
    noise_sd: float = 3.0,
    random_state: Any = 0,
    data_home: str | os.PathLike | None = None,
) -> GraphDataset:
    """A concert promotion on the real LastFM Asia social network (semi-synthetic).

    The network (7,624 users, 27,806 mutual follows) and covariates (activity, taste) are
    real; the promotion ``promo`` and outcome ``listening_hours`` are simulated, so the true
    effects are known. Exposure ``"mean"``: the share of friends who received the promo.
    The targeting rule and the outcome both depend nonlinearly on friends' taste.
    """
    rng = check_random_state(random_state)
    net, X, names = lastfm_covariates(data_home)
    A = net.adjacency
    P = A.copy()
    P.data[:] = 1.0
    M = row_normalize(P)
    a, s1, s2, s3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    nbr_s1, nbr_a = M @ s1, M @ a
    max_s2 = _aggregate(P, M, s2[:, None], "max").ravel()

    logit = -0.3 + 0.6 * s1 + 0.4 * a + 0.3 * s2 * s3 + confounding * (0.8 * nbr_s1 + 0.4 * nbr_a)
    T = rng.binomial(1, expit(logit)).astype(float)
    Y = (
        20.0
        + 3.0 * a
        + 2.0 * s1
        + 1.5 * np.sin(2 * s2)
        + 1.0 * s3**2
        + confounding * (2.5 * nbr_s1 + 1.5 * max_s2)
        + direct_effect * T
        + peer_effect * (M @ T)
        + noise_sd * rng.standard_normal(len(T))
    )
    data = GraphData(X, T, Y, A, feature_names=names)
    return GraphDataset(
        name="lastfm_promo",
        data=data,
        truth={"ade": direct_effect, "ape": peer_effect},
        exposure="mean",
        treatment_name="promo",
        outcome_name="listening_hours",
        DESCR=load_descr("lastfm_promo"),
        extras={"country": net.attributes["country"], "citation": net.citation},
    )
