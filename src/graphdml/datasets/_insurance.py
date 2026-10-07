"""Cai, de Janvry & Sadoulet (2015): a randomized experiment on a real friendship network.

Rice farmers in rural China were randomly assigned to simple or intensive information
sessions about weather insurance, in two rounds three days apart. Each household named up
to five friends. The study finds that intensive sessions raise one's own take-up (direct
effect) and that second-round farmers are more likely to buy when more of their friends
attended a first-round intensive session (peer effect).

The replication package is CC BY 4.0 (data) and is available from openICPSR, project
113593 (free account required). Pass the folder containing ``0422survey.dta`` and
``0422allinforawnet.dta``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graphdml.data import GraphData
from graphdml.datasets._base import GraphDataset, load_descr
from graphdml.exposure import MatrixExposure

__all__ = ["load_insurance_experiment"]

CITATION = (
    "Cai, J., de Janvry, A. and Sadoulet, E. (2015). Social Networks and the Decision to "
    "Insure. American Economic Journal: Applied Economics 7(2): 81-108. Data: openICPSR "
    "project 113593, https://doi.org/10.3886/E113593V1 (CC BY 4.0)."
)

#: Estimates reported in the study's Table 2 (OLS with village fixed effects, SEs
#: clustered by natural village).
PUBLISHED = {
    "ade_first_round": {"coef": 0.1408, "std_err": 0.0260, "n": 2137},
    "ape_second_round": {"coef": 0.2912, "std_err": 0.0820, "n": 1255},
    "ade_second_round": {"coef": 0.0298, "std_err": 0.0332, "n": 1255},
}

_COVARIATES = ["male", "age", "agpop", "ricearea_2010", "literacy", "risk_averse",
               "disaster_prob"]

_REQUIRED_FILES = ("0422survey.dta", "0422allinforawnet.dta")
_DATA_URL = "https://www.openicpsr.org/openicpsr/project/113593/version/V1/view"


def load_insurance_experiment(data_dir: str | os.PathLike) -> GraphDataset:
    """Load the insurance experiment as a :class:`GraphDataset` (no ground truth).

    * Nodes: the 4,902 surveyed households. An edge ``j -> i`` means ``i`` named ``j`` as
      a friend (``j`` can influence ``i``).
    * ``T`` = assigned to an intensive session (randomized); ``Y`` = bought insurance.
    * Covariates: household characteristics (median-imputed, with missing indicators),
      the randomized design variables (second round, information treatments), the number
      of friends named, and administrative-village dummies.
    * ``extras["exposure"]``: the study's peer exposure, the share of named friends who
      attended a *first-round intensive* session (the denominator counts every named
      friend, surveyed or not, as in the study). It is linear in ``T``, so it is given as
      a :class:`~graphdml.MatrixExposure`.
    * ``extras`` also has ``second_round``, ``info_none``, ``natural_village`` (for
      clustered SEs), the study's own exposure variable for checking, and the published
      estimates.
    """
    data_dir = Path(data_dir).expanduser()
    missing = [f for f in _REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} in {data_dir}. This dataset is not bundled "
            "with graphdml: download the replication package (free openICPSR account, CC BY "
            f"4.0) from {_DATA_URL}, unzip it, and pass the folder that contains the .dta "
            "files (the 'data/data' folder)."
        )
    survey = pd.read_stata(data_dir / "0422survey.dta")
    noms = pd.read_stata(data_dir / "0422allinforawnet.dta")
    noms = noms[~((noms["network_missname"] == 1) & noms["network_id"].isna())]

    survey = survey.sort_values("id").reset_index(drop=True)
    n = len(survey)
    pos = pd.Series(np.arange(n), index=survey["id"].astype(int))

    # The study's exposure, computed exactly as in rawnet.do (for checking our operator).
    n_named = noms.groupby("id").size()
    first_int = ((noms["delay"] == 0) & (noms["intensive"] == 1)).groupby(noms["id"]).sum()
    published_rate = (first_int / n_named).reindex(survey["id"]).to_numpy()

    inside = noms[noms["network_id"].isin(pos.index) & noms["id"].isin(pos.index)]
    rows = pos[inside["id"].astype(int)].to_numpy()  # nominator i (influenced)
    cols = pos[inside["network_id"].astype(int)].to_numpy()  # named friend j
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    A = sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0

    named = n_named.reindex(survey["id"]).fillna(0).to_numpy()
    first_round = (survey["delay"] == 0).to_numpy()
    inv_named = np.divide(1.0, named, out=np.zeros(n), where=named > 0)
    E = sp.csr_array(sp.diags_array(inv_named) @ A @ sp.diags_array(first_round.astype(float)))

    X_parts, names = [], []
    for c in _COVARIATES:
        v = survey[c].astype(float).to_numpy()
        miss = np.isnan(v)
        X_parts.append(np.where(miss, np.nanmedian(v), v))
        names.append(c)
        if miss.any():
            X_parts.append(miss.astype(float))
            names.append(f"{c}_missing")
    design = {
        "second_round": survey["delay"].to_numpy(dtype=float),
        "info_takeup_rate": survey["info_takeup_rate"].fillna(0).to_numpy(dtype=float),
        "info_takeup_list": survey["info_takeup_list"].fillna(0).to_numpy(dtype=float),
        "friends_named": named,
    }
    for k, v in design.items():
        X_parts.append(v)
        names.append(k)
    villages = pd.get_dummies(survey["village"], prefix="village", dtype=float)
    X = np.column_stack([*X_parts, villages.to_numpy()])
    names += list(villages.columns)

    data = GraphData(
        X,
        survey["intensive"].to_numpy(dtype=float),
        survey["takeup_survey"].to_numpy(dtype=float),
        A,
        feature_names=names,
        node_ids=survey["id"].astype(int).to_numpy(),
        directed=True,
    )
    exposure = MatrixExposure(
        E,
        name="friends_first_round_intensive",
        units="change in take-up going from 0% to 100% of named friends in a first-round "
        "intensive session",
    )
    return GraphDataset(
        name="insurance_experiment",
        data=data,
        truth={},
        exposure="friends_first_round_intensive",
        treatment_name="intensive_session",
        outcome_name="bought_insurance",
        DESCR=load_descr("insurance_experiment"),
        extras={
            "exposure": exposure,
            "second_round": ~first_round,
            "info_none": (survey["info_none"] == 1).to_numpy(),
            "natural_village": survey["address"].to_numpy(),
            "published_exposure": published_rate,
            "published": PUBLISHED,
            "citation": CITATION,
        },
    )
