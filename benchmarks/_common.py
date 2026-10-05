"""Shared helpers for the Monte Carlo benchmarks."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def run_seeds(fn: Callable[[int], list[dict]], seeds: range, n_jobs: int = -1) -> pd.DataFrame:
    """Run ``fn(seed)`` (returning a list of row dicts) in parallel, one thread per worker."""

    def wrapped(seed: int) -> list[dict]:
        os.environ["OMP_NUM_THREADS"] = "1"
        with threadpool_limits(1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(seed)

    rows = Parallel(n_jobs=n_jobs)(delayed(wrapped)(s) for s in seeds)
    return pd.DataFrame([r for chunk in rows for r in chunk])


def estimate_rows(model, truth: dict, seed: int, **labels) -> list[dict]:
    """One row per effect: estimate, SE, CI and whether it covers the truth."""
    frame = model.summary_frame()
    keys = {"direct": "ade"}
    rows = []
    for name, r in frame.iterrows():
        key = keys.get(name, "ape")
        t = truth[key]
        rows.append(
            {
                **labels,
                "seed": seed,
                "effect": "direct" if name == "direct" else "peer",
                "truth": t,
                "estimate": r["coef"],
                "se": r["std_err"],
                "covers": r["ci_lower"] <= t <= r["ci_upper"],
                "n_focal": model.n_focal_,
            }
        )
    return rows


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Bias, RMSE, coverage, mean SE / empirical SD (calibration) per group."""

    def agg(g: pd.DataFrame) -> pd.Series:
        err = g["estimate"] - g["truth"]
        sd = g["estimate"].std(ddof=1)
        return pd.Series(
            {
                "truth": g["truth"].iloc[0],
                "mean_est": g["estimate"].mean(),
                "bias": err.mean(),
                "rel_bias": err.mean() / abs(g["truth"].iloc[0]) if g["truth"].iloc[0] else np.nan,
                "rmse": np.sqrt((err**2).mean()),
                "coverage": g["covers"].mean(),
                "se_over_sd": g["se"].mean() / sd if sd > 0 else np.nan,
                "n_focal": g["n_focal"].mean(),
                "reps": len(g),
            }
        )

    return df.groupby(by, sort=False).apply(agg, include_groups=False).reset_index()


def write(df: pd.DataFrame, summary: pd.DataFrame, name: str, title: str, note: str = "") -> str:
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    md = f"## {title}\n\n{note}\n\n" + summary.to_markdown(index=False, floatfmt=".3f") + "\n"
    (RESULTS / f"{name}.md").write_text(md)
    return md
