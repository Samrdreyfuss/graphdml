"""Which defaults should GraphDML ship with? Evidence for questions 1a, 1b and 1c.

Compares, with the default machine-learning nuisances (regularised gradient boosting):

* score: IV-type (default) vs partialling-out (paper)                        -> question 1b
* nuisance training: buffered vs focal-only (paper)                 -> question 1a
* featurizer hops: 2 (default) vs 1 (paper's one-layer GNN)         -> question 1c
* paper mode: the published procedure end to end

    python benchmarks/design_choices.py --seeds 40
"""

from __future__ import annotations

import argparse

from _common import estimate_rows, run_seeds, summarize, write

from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.datasets import make_flu_town, make_referral_app
from graphdml.simulate import erdos_renyi, make_paper_linear, make_paper_nonlinear

CONFIGS = {
    "default (IV-type, buffered, 2 hops)": {},
    "partialling-out score": {"score": "partialling-out"},
    "focal-only training": {"nuisance_training": "focal"},
    "1-hop features": {"featurizer": NeighborhoodFeatures(hops=1)},
    "partialling-out + 1-hop": {"score": "partialling-out",
                                "featurizer": NeighborhoodFeatures(hops=1)},
    "paper mode": {"mode": "paper"},
}

DATASETS = {
    "flu_town (n=3000)": lambda s: make_flu_town(3000, random_state=s),
    "referral_app (n=4000)": lambda s: make_referral_app(4000, random_state=s),
    "paper eq. 38, ER n=3000": lambda s: make_paper_linear(
        erdos_renyi(3000, 4.0, random_state=s), noise_sd=1.0, random_state=s
    ),
    "paper eq. 39, ER n=3000": lambda s: make_paper_nonlinear(
        erdos_renyi(3000, 4.0, random_state=s), noise_sd=1.0, random_state=s
    ),
}


def one_seed(seed: int) -> list[dict]:
    rows = []
    for dname, make in DATASETS.items():
        ds = make(seed)
        for cname, kw in CONFIGS.items():
            m = GraphDML(exposure=ds.exposure, random_state=seed, **kw).fit(ds.data)
            rows += estimate_rows(m, ds.truth, seed, dataset=dname, config=cname)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40)
    args = p.parse_args()
    df = run_seeds(one_seed, range(args.seeds))
    summary = summarize(df, ["dataset", "effect", "config"])
    print(write(df, summary, "design_choices", "Design choices with ML nuisances",
                f"{args.seeds} seeds per dataset; default learners (regularised HGB)."))
