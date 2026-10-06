"""Does the total effect (ATE: treat every node vs none) cover at the nominal rate?

The ATE is direct + sum_k peer_k * (average exposure k when all nodes are treated). Its
standard error combines the direct and peer estimates through their covariance.

    python benchmarks/ate_coverage.py --seeds 300
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import run_seeds, write
from sklearn.linear_model import LinearRegression

from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.datasets import make_flu_town, make_referral_app
from graphdml.simulate import barabasi_albert, erdos_renyi, make_linear_gaussian, oracle_learners

LINEAR = {
    "featurizer": NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
    "model_y": LinearRegression(),
    "model_t": LinearRegression(),
}


def _row(m, truth_ate, seed, **labels):
    f = m.summary_frame().loc["total"]
    return {**labels, "seed": seed, "truth": truth_ate, "estimate": f["coef"],
            "se": f["std_err"], "covers": f["ci_lower"] <= truth_ate <= f["ci_upper"]}


def one_seed(seed: int) -> list[dict]:
    rows = []
    for gname, A in [("ER", erdos_renyi(2000, 4.0, random_state=seed)),
                     ("BA (hubs)", barabasi_albert(2000, 2, random_state=seed))]:
        for exposure in ["mean", "sum"]:
            ds = make_linear_gaussian(A, theta=1.0, alpha=0.5, exposure=exposure,
                                      random_state=seed)
            for nz, kw in [("oracle", oracle_learners(ds)), ("linear (correct)", LINEAR)]:
                if nz == "linear (correct)" and exposure == "sum":
                    continue  # linear features are not exact for the sum exposure
                m = GraphDML(**kw, exposure=exposure, random_state=seed).fit(ds.data)
                truth = 1.0 + 0.5 * m.total_weights_[1]
                rows.append(_row(m, truth, seed, setting=f"{gname}, {exposure}, {nz}"))
    if seed < 100:  # ML nuisances on the story datasets (slower)
        for name, ds in [("flu_town (n=3000), HGB", make_flu_town(3000, random_state=seed)),
                         ("referral_app (n=4000), HGB",
                          make_referral_app(4000, random_state=seed))]:
            m = GraphDML(exposure=ds.exposure, random_state=seed).fit(ds.data)
            truth = ds.truth["ade"] + ds.truth["ape"] * m.total_weights_[1]
            rows.append(_row(m, truth, seed, setting=name))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=300)
    args = p.parse_args()
    df = run_seeds(one_seed, range(args.seeds))
    g = df.groupby("setting", sort=False)
    summary = g.agg(reps=("seed", "size"), truth=("truth", "mean"), mean_est=("estimate", "mean"),
                    coverage=("covers", "mean"), mean_se=("se", "mean"),
                    sd=("estimate", "std")).reset_index()
    summary["rel_bias"] = (summary["mean_est"] - summary["truth"]) / summary["truth"].abs()
    summary["se_over_sd"] = summary["mean_se"] / summary["sd"]
    summary["band"] = [f"0.95 ± {3 * np.sqrt(0.95 * 0.05 / r):.3f}" for r in summary["reps"]]
    cols = ["setting", "reps", "truth", "mean_est", "rel_bias", "coverage", "se_over_sd", "band"]
    print(write(df, summary[cols], "ate_coverage", "Total effect (ATE) coverage",
                "Truth = direct + peer × average exposure when all nodes are treated."))
