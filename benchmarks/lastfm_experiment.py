"""Experiment on a real network: semi-synthetic promotion on LastFM Asia.

The graph (7,624 users) and covariates are real; treatment and outcome are simulated
anew in each replication, so the true effects are known (direct 2.0, peer 3.0). Compares
GraphDML (default and original procedure) with methods that ignore the network.

    python benchmarks/lastfm_experiment.py --seeds 100
"""

from __future__ import annotations

import argparse

from _common import estimate_rows, run_seeds, summarize, write

from graphdml import GraphDML
from graphdml.baselines import aggregate_dml, iid_dml, naive_ols
from graphdml.datasets import make_lastfm_promo


def _ols_rows(frame, truth, seed, method):
    rows = []
    for name, r in frame.iterrows():
        effect = "direct" if name == "direct" else "peer"
        t = truth["ade" if effect == "direct" else "ape"]
        rows.append({"method": method, "seed": seed, "effect": effect, "truth": t,
                     "estimate": r["coef"], "se": r["std_err"],
                     "covers": r["ci_lower"] <= t <= r["ci_upper"], "n_focal": float("nan")})
    return rows


def one_seed(seed: int) -> list[dict]:
    ds = make_lastfm_promo(random_state=seed)
    d, truth = ds.data, ds.truth
    rows = []
    models = {
        "GraphDML (default)": GraphDML(exposure="mean", random_state=seed),
        "GraphDML (random-order focal set)": GraphDML(exposure="mean", focal_set="random",
                                                      random_state=seed),
        "GraphDML (original procedure)": GraphDML(exposure="mean", mode="original",
                                                  random_state=seed),
    }
    for method, m in models.items():
        rows += estimate_rows(m.fit(d), truth, seed, method=method)
    rows += estimate_rows(aggregate_dml(d, "mean", random_state=seed), truth, seed,
                          method="DML + 1-hop aggregates, all nodes")
    rows += estimate_rows(iid_dml(d, random_state=seed), truth, seed,
                          method="DML, own covariates (i.i.d.)")
    rows += _ols_rows(naive_ols(d, "mean"), truth, seed, "OLS + exposure (no network)")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    args = p.parse_args()
    make_lastfm_promo()  # download and cache before forking workers
    df = run_seeds(one_seed, range(args.seeds))
    summary = summarize(df, ["effect", "method"])
    print(write(df, summary, "lastfm_experiment",
                "Semi-synthetic experiment on the LastFM Asia network",
                f"{args.seeds} replications; real graph and covariates, simulated promo and "
                "listening hours; truth: direct 2.0, peer 3.0 (exposure: share of friends)."))
