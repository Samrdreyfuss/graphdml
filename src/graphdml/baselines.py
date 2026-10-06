"""Simpler estimators to compare GraphDML against.

These exist to show *why* the network matters: on data with network confounding they are
biased, and the tutorials put them side by side with GraphDML.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from graphdml.data import GraphData
from graphdml.estimator import GraphDML
from graphdml.exceptions import GraphDMLWarning
from graphdml.exposure import resolve_exposures
from graphdml.features import NeighborhoodFeatures
from graphdml.inference import normal_ci, ols_hc0

__all__ = ["aggregate_dml", "compare_methods", "iid_dml", "naive_ols"]


def naive_ols(data: GraphData, exposure: Any = None, alpha: float = 0.05) -> pd.DataFrame:
    """OLS of Y on T, own covariates and (optionally) exposure, with HC0 errors.

    Ignores network confounding: neighbors' covariates are left out.
    """
    exposures = resolve_exposures(exposure)
    Z = [e(data) for e in exposures]
    D = np.column_stack([np.ones(data.n_nodes), data.T, *Z, data.X])
    coef, cov = ols_hc0(D, data.Y)
    k = 1 + len(Z)
    c, se = coef[1 : 1 + k], np.sqrt(np.diag(cov))[1 : 1 + k]
    lo, hi = normal_ci(c, se, alpha)
    names = ["direct"] + [f"peer:{e.name}" for e in exposures]
    return pd.DataFrame(
        {"coef": c, "std_err": se, "ci_lower": lo, "ci_upper": hi},
        index=pd.Index(names, name="effect"),
    )


def iid_dml(data: GraphData, **kwargs: Any) -> GraphDML:
    """Standard DML that ignores the network: own covariates, no exposure, all nodes."""
    params = {"featurizer": "own", "exposure": None, "focal_set": None, **kwargs}
    return GraphDML(**params).fit(data)


def aggregate_dml(data: GraphData, exposure: Any = "sum", **kwargs: Any) -> GraphDML:
    """DML with predefined one-hop neighbor aggregates and no focal set.

    It adjusts for neighbors' covariates but treats all nodes as independent.
    """
    params = {
        "featurizer": NeighborhoodFeatures(aggs=("mean", "max", "min"), hops=1),
        "exposure": exposure,
        "focal_set": None,
        **kwargs,
    }
    return GraphDML(**params).fit(data)


def compare_methods(
    data: GraphData,
    exposure: Any = "sum",
    truth: dict[str, float] | None = None,
    alpha: float = 0.05,
    random_state: int = 0,
    estimator: GraphDML | None = None,
) -> pd.DataFrame:
    """Fit the baselines and GraphDML; return one tidy table (with truth if known)."""
    frames = {
        "Naive OLS (no network)": naive_ols(data, alpha=alpha),
        "OLS + exposure (no neighbor covariates)": naive_ols(data, exposure, alpha=alpha),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GraphDMLWarning)
        frames["DML, own covariates (i.i.d.)"] = iid_dml(
            data, random_state=random_state
        ).summary_frame(alpha, total=False)
        frames["DML + neighbor aggregates (no focal set)"] = aggregate_dml(
            data, exposure, random_state=random_state
        ).summary_frame(alpha, total=False)
        est = estimator if estimator is not None else GraphDML(
            exposure=exposure, random_state=random_state
        )
        frames["GraphDML"] = est.fit(data).summary_frame(alpha, total=False)

    rows = []
    for method, frame in frames.items():
        for name, row in frame.iterrows():
            effect = "direct" if name == "direct" else "peer"
            rows.append(
                {
                    "method": method,
                    "effect": effect,
                    "estimate": row["coef"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                }
            )
    table = pd.DataFrame(rows)
    if truth:
        key = {"direct": "ade", "peer": "ape"}
        table["truth"] = table["effect"].map(lambda e: truth.get(key[e], np.nan))
        table["covers_truth"] = (table["ci_lower"] <= table["truth"]) & (
            table["truth"] <= table["ci_upper"]
        )
    return table
