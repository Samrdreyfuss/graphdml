"""GraphDML on a real randomized network experiment (Cai, de Janvry & Sadoulet 2015).

Compares GraphDML's direct and peer estimates with the published OLS estimates. Needs the
openICPSR replication data (project 113593):

    python benchmarks/insurance_experiment.py --data path/to/113593-V1/data/data
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from _common import RESULTS

from graphdml import GraphDML
from graphdml.datasets import load_insurance_experiment

SEEDS = range(5)


def fit(d, villages, label, **kw) -> list[dict]:
    fits = [GraphDML(random_state=s, **kw).fit(d) for s in SEEDS]
    rows = []
    for j, name in enumerate(fits[0].coef_names_):
        coefs = [m.coef_[j] for m in fits]
        rows.append({
            "analysis": label,
            "effect": name,
            "estimate (median of 5 seeds)": np.median(coefs),
            "se": np.median([m.se_[j] for m in fits]),
            "se clustered by village": np.median(
                [m.cluster_summary_frame(villages)["std_err"].iloc[j] for m in fits]),
            "seed spread": np.ptp(coefs),
            "focal n": fits[0].n_focal_,
        })
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    warnings.simplefilter("ignore")
    ds = load_insurance_experiment(args.data)
    d, ex = ds.data, ds.extras
    E = ex["exposure"]
    agree = np.isclose(E(d), np.nan_to_num(ex["published_exposure"], nan=-1))
    peer_pop = ex["second_round"] & ex["info_none"] & ~np.isnan(ex["published_exposure"])
    v = ex["natural_village"]
    rows = []
    rows += fit(d, v, "first round", exposure=None, estimation_nodes=~ex["second_round"])
    rows += fit(d, v, "second round, no info", exposure=E, estimation_nodes=peer_pop)
    rows += fit(d, v, "second round, no info, exposures agree", exposure=E,
                estimation_nodes=peer_pop & agree)
    table = pd.DataFrame(rows)
    pub = ex["published"]
    md = "\n".join([
        "## Real randomized experiment: insurance take-up (Cai et al. 2015)",
        "",
        f"Published (OLS, village FE, clustered SEs): direct, first round "
        f"{pub['ade_first_round']['coef']} ({pub['ade_first_round']['std_err']}); peer, "
        f"second round {pub['ape_second_round']['coef']} ({pub['ape_second_round']['std_err']}).",
        "",
        table.to_markdown(index=False, floatfmt=".3f"),
    ])
    (RESULTS / "insurance_experiment.md").write_text(md + "\n")
    print(md)
