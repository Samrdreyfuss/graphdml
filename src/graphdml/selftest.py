"""``graphdml.selftest()``: check on your own machine that the estimator finds real signals
and does not invent false ones.

It repeatedly simulates two worlds with the same strong network confounding, one with
planted effects and one with none (the *null twin*), fits graphdml, and compares the
estimates with the known truth. Methods that ignore the network are shown alongside, to
show what the confounding would do to them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from graphdml.baselines import naive_ols
from graphdml.datasets._signal import make_signal_check
from graphdml.estimator import GraphDML
from graphdml.exceptions import GraphDMLWarning

__all__ = ["SelfTestResult", "selftest"]

_EFFECTS = ("direct", "peer", "total")


@dataclass
class SelfTestResult:
    """Outcome of :func:`selftest`. ``passed`` is True when every check below holds."""

    passed: bool
    table: pd.DataFrame
    checks: pd.DataFrame
    n_reps: int

    def __repr__(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        return (
            f"graphdml self-test: {verdict} ({self.n_reps} repetitions per world)\n\n"
            f"{self.table.to_string(float_format=lambda v: f'{v:.3f}')}\n\n"
            f"{self.checks.to_string()}"
        )


def _one_rep(seed, n, direct_effect, peer_effect, estimator, alpha):
    rows = []
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", GraphDMLWarning)
        worlds = {"planted": (direct_effect, peer_effect), "null": (0.0, 0.0)}
        for world, (ade, ape) in worlds.items():
            ds = make_signal_check(n, direct_effect=ade, peer_effect=ape, random_state=seed)
            model = clone(estimator) if estimator is not None else GraphDML(exposure="mean")
            model.set_params(random_state=seed).fit(ds.data)
            frame = model.summary_frame(alpha)
            naive = naive_ols(ds.data, "mean", alpha)
            truth = {"direct": ade, "peer": ape, "total": ade + ape * model.total_weights_[1]}
            for key, name in zip(_EFFECTS, ("direct", "peer:mean", "total"), strict=True):
                r = frame.loc[name]
                row = {"world": world, "effect": key, "truth": truth[key], "estimate": r["coef"],
                       "covers": r["ci_lower"] <= truth[key] <= r["ci_upper"],
                       "detected": r["p_value"] < alpha,
                       "right_sign": np.sign(r["coef"]) == np.sign(truth[key]) or truth[key] == 0}
                if key != "total":
                    z = naive.loc[name, "coef"] / naive.loc[name, "std_err"]
                    row["naive_detected"] = float(2 * stats.norm.sf(abs(z)) < alpha)
                rows.append(row)
    return rows


def selftest(
    n_reps: int = 30,
    n: int = 6000,
    *,
    direct_effect: float = 1.0,
    peer_effect: float = 0.6,
    estimator: GraphDML | None = None,
    alpha: float = 0.05,
    n_jobs: int = -1,
    random_state: int = 0,
) -> SelfTestResult:
    """Run the signal check and report whether graphdml passed.

    Parameters
    ----------
    n_reps : repetitions per world; each fits one model on a fresh ``n``-node graph.
    estimator : an unfitted :class:`GraphDML` to test (default: ``GraphDML(exposure="mean")``
        with the library's default learners). Pass your own to test your settings.

    Checks (set from statistics, not tuned to results; the bands use 3 Monte Carlo
    standard errors, so they loosen when ``n_reps`` is small):

    * **coverage**: the share of 95% intervals containing the truth is at least
      ``0.95 - 3 * SE`` in the planted world, for every effect;
    * **power**: the planted direct and peer effects are detected in at least 80% of runs,
      with the correct sign;
    * **false alarms**: in the null world, the share of runs that claim an effect is at
      most ``alpha + 3 * SE``, for every effect.
    """
    seeds = range(random_state * 10_000, random_state * 10_000 + n_reps)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_rep)(s, n, direct_effect, peer_effect, estimator, alpha) for s in seeds
    )
    df = pd.DataFrame([r for chunk in rows for r in chunk])
    g = df.groupby(["world", "effect"], sort=False)
    table = g.agg(truth=("truth", "mean"), mean_estimate=("estimate", "mean"),
                  coverage=("covers", "mean"), detected=("detected", "mean"),
                  naive_detected=("naive_detected", "mean")).reset_index()
    table = table.set_index(["world", "effect"])

    se = np.sqrt(0.95 * 0.05 / n_reps)
    se_fp = np.sqrt(alpha * (1 - alpha) / n_reps)
    checks = []
    for effect in _EFFECTS:
        cov = table.loc[("planted", effect), "coverage"]
        checks.append((f"planted {effect}: interval coverage", cov, f">= {0.95 - 3 * se:.2f}",
                       cov >= 0.95 - 3 * se))
    for effect in ("direct", "peer"):
        power = df[(df.world == "planted") & (df.effect == effect)]
        rate = float((power.detected & power.right_sign).mean())
        label = f"planted {effect}: detected with right sign"
        checks.append((label, rate, ">= 0.80", rate >= 0.80))
    for effect in _EFFECTS:
        fp = table.loc[("null", effect), "detected"]
        checks.append((f"null {effect}: false alarms", fp, f"<= {alpha + 3 * se_fp:.2f}",
                       fp <= alpha + 3 * se_fp))
    columns = ["check", "value", "required", "ok"]
    checks_df = pd.DataFrame(checks, columns=columns).set_index("check")
    checks_df["value"] = checks_df["value"].map("{:.2f}".format)
    return SelfTestResult(bool(checks_df["ok"].all()), table, checks_df, n_reps)
