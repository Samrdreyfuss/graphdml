"""Layer 3 gate: do 95% intervals cover at the nominal rate?

Runs GraphDML on the linear DGP across topologies with (a) oracle nuisances and (b)
correctly specified linear nuisances, and on the paper's binary-treatment DGP with oracle
nuisances. Pass criterion (VALIDATION.md): coverage within 0.95 ± 3 Monte Carlo standard
errors in every row.

    python benchmarks/coverage.py --seeds 300
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import estimate_rows, run_seeds, summarize, write
from sklearn.linear_model import LinearRegression

from graphdml import GraphDML, NeighborhoodFeatures
from graphdml.simulate import (
    barabasi_albert,
    erdos_renyi,
    make_linear_gaussian,
    make_paper_linear,
    oracle_learners,
    stochastic_block_model,
)

N = 2000
GRAPHS = {
    "ER (avg deg 4)": lambda s: erdos_renyi(N, 4.0, random_state=s),
    "BA (m=2, hubs)": lambda s: barabasi_albert(N, 2, random_state=s),
    "SBM (40 blocks)": lambda s: stochastic_block_model([50] * 40, 0.1, 0.001, random_state=s),
}


LINEAR = "linear, continuous T"
PAPER = "paper eq. 38, binary T"
CELLS = [(g, d, nz) for g in GRAPHS for d, nz in
         [(LINEAR, "oracle"), (LINEAR, "linear (correct)"), (PAPER, "oracle")]]


def one_seed(seed: int, cells=None) -> list[dict]:
    rows = []
    for gname, dgp, nuisances in cells or CELLS:
        A = GRAPHS[gname](seed)
        if dgp == LINEAR:
            ds = make_linear_gaussian(A, theta=1.0, alpha=0.5, random_state=seed)
            exposure = "mean"
        else:
            ds = make_paper_linear(A, noise_sd=1.0, random_state=seed)
            exposure = "sum"
        if nuisances == "oracle":
            m = GraphDML(**oracle_learners(ds), exposure=exposure, random_state=seed)
        else:
            m = GraphDML(
                LinearRegression(), LinearRegression(),
                featurizer=NeighborhoodFeatures(aggs=("mean",), hops=2, include_degree=False),
                exposure=exposure, random_state=seed,
            )
        rows += estimate_rows(m.fit(ds.data), ds.truth, seed, graph=gname, dgp=dgp,
                              nuisances=nuisances)
    return rows


def band(n_seeds: int) -> float:
    return 3 * np.sqrt(0.95 * 0.05 / n_seeds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=300)
    p.add_argument("--confirm-seeds", type=int, default=2000)
    args = p.parse_args()
    by = ["dgp", "nuisances", "graph", "effect"]

    # Stage 1: screen every cell.
    df = run_seeds(one_seed, range(args.seeds))
    summary = summarize(df, by)
    half = band(args.seeds)
    summary["in_band"] = summary["coverage"].between(0.95 - half, 0.95 + half)
    note = (f"Stage 1 (screen): {args.seeds} seeds, n = {N} nodes, nominal 95% intervals. "
            f"Band 0.95 ± {half:.3f} (3 Monte Carlo SEs).")

    # Stage 2: rerun out-of-band cells with fresh seeds and a tighter band.
    flagged = summary.loc[~summary["in_band"], ["graph", "dgp", "nuisances"]]
    cells = sorted({tuple(r) for r in flagged.to_numpy()})
    if cells:
        seeds = range(10_000, 10_000 + args.confirm_seeds)
        conf = run_seeds(lambda s: one_seed(s, cells), seeds)
        conf_summary = summarize(conf, by)
        h2 = band(args.confirm_seeds)
        conf_summary["in_band"] = conf_summary["coverage"].between(0.95 - h2, 0.95 + h2)
        write(conf, conf_summary, "coverage_confirm", "Coverage gate, stage 2 (confirmation)",
              f"Cells outside the stage-1 band, rerun with {args.confirm_seeds} fresh seeds. "
              f"Band 0.95 ± {h2:.3f}.")
        note += f" Stage 2 reran {len(cells)} flagged cell(s): see coverage_confirm.md."
    print(write(df, summary, "coverage", "Coverage gate", note))
