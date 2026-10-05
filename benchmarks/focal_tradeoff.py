"""What does the focal set cost? GraphDML vs the same estimator on all nodes.

Without a focal set, every node contributes but neighbors' scores are dependent and the
i.i.d. variance formula ignores that. With a focal set, intervals are valid but only a
fraction of nodes contribute.

    python benchmarks/focal_tradeoff.py --seeds 100
"""

from __future__ import annotations

import argparse

from _common import estimate_rows, run_seeds, summarize, write

from graphdml import GraphDML
from graphdml.baselines import aggregate_dml
from graphdml.datasets import make_flu_town, make_referral_app

DATASETS = {
    "flu_town (n=3000)": lambda s: make_flu_town(3000, random_state=s),
    "referral_app (n=4000)": lambda s: make_referral_app(4000, random_state=s),
}


def one_seed(seed: int) -> list[dict]:
    rows = []
    for name, make in DATASETS.items():
        ds = make(seed)
        fits = {
            "GraphDML (focal set)": GraphDML(exposure=ds.exposure, random_state=seed),
            "GraphDML features, all nodes": GraphDML(exposure=ds.exposure, focal_set=None,
                                                     random_state=seed),
        }
        for method, model in fits.items():
            rows += estimate_rows(model.fit(ds.data), ds.truth, seed, dataset=name,
                                  method=method)
        pa = aggregate_dml(ds.data, ds.exposure, random_state=seed)
        rows += estimate_rows(pa, ds.truth, seed, dataset=name,
                              method="1-hop aggregates, all nodes")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    args = p.parse_args()
    df = run_seeds(one_seed, range(args.seeds))
    summary = summarize(df, ["dataset", "effect", "method"])
    print(write(df, summary, "focal_tradeoff", "The cost of the focal set",
                f"{args.seeds} seeds; default learners."))
