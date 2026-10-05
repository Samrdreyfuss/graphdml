"""Which defaults should GraphDML ship with?

Compares, with the default machine-learning nuisances (regularised gradient boosting):

* score: IV-type (default) vs partialling-out
* nuisance training: buffered vs focal-only
* featurizer hops: 2 (default) vs 1
* original procedure: mode="original" end to end

    python benchmarks/design_choices.py --seeds 40
"""

from __future__ import annotations

import argparse

from _common import estimate_rows, run_seeds, summarize, write

from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.datasets import make_flu_town, make_referral_app
from graphdml.simulate import erdos_renyi, make_benchmark_linear, make_benchmark_nonlinear

CONFIGS = {
    "default (IV-type, buffered, 2 hops)": {},
    "partialling-out score": {"score": "partialling-out"},
    "focal-only training": {"nuisance_training": "focal"},
    "1-hop features": {"featurizer": NeighborhoodFeatures(hops=1)},
    "partialling-out + 1-hop": {"score": "partialling-out",
                                "featurizer": NeighborhoodFeatures(hops=1)},
    "original procedure": {"mode": "original"},
}

DATASETS = {
    "flu_town (n=3000)": lambda s: make_flu_town(3000, random_state=s),
    "referral_app (n=4000)": lambda s: make_referral_app(4000, random_state=s),
    "benchmark linear, ER n=3000": lambda s: make_benchmark_linear(
        erdos_renyi(3000, 4.0, random_state=s), noise_sd=1.0, random_state=s
    ),
    "benchmark nonlinear, ER n=3000": lambda s: make_benchmark_nonlinear(
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
