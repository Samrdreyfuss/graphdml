"""Plain-text summaries of fitted GraphDML estimators."""

from __future__ import annotations

import html
import textwrap
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from graphdml.estimator import GraphDML

__all__ = ["Summary", "format_summary"]

_WIDTH = 78


class Summary:
    """Text summary that renders nicely in a terminal and in notebooks."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(self.text)}</pre>"


def format_summary(est: GraphDML, alpha: float = 0.05) -> Summary:
    s, d = est.settings_, est.diagnostics_
    frame = est.summary_frame(alpha)
    binary = est.treatment_is_binary_
    lines: list[str] = []
    add = lines.append

    add("GraphDML: treatment effects on a network")
    add("═" * _WIDTH)
    exposure = ", ".join(e.name for e in est.exposures_) or "none (direct effect only)"
    add(
        f"Treatment: {'binary' if binary else 'continuous'}   Exposure: {exposure}   "
        f"Score: {s['score']}"
    )
    add(
        f"Folds: {s['n_folds']} × {est.n_rep} rep   Aggregation: {s['aggregation']}   "
        f"Nuisance training: {s['nuisance_training']}"
    )
    focal_label = "all nodes (no focal set)" if s["focal_set"] is None else (
        f"{d['n_focal']:,} ({d['focal_share']:.1%})"
    )
    add(f"Nodes: {d['n_nodes']:,}   Focal nodes (effective sample size): {focal_label}")
    add("")

    # Practitioner-facing labels and units; the ATE (total effect) comes first.
    labels = {"total": "ATE (total)", "direct": "direct (ADE)"}
    units = {
        "total": (
            "change in Y from treating every node vs none (direct + peer)"
            if binary
            else "change in Y from raising every node's treatment by one unit"
        ),
        "direct": (
            "change in Y when own treatment goes 0 → 1, neighbors unchanged"
            if binary
            else "change in Y per unit of own treatment, neighbors unchanged"
        ),
    }
    for name, e in zip(est.coef_names_[1:], est.exposures_, strict=True):
        labels[name] = f"peer (APE): {e.name}"
        units[name] = e.units(binary)
    order = [n for n in ["total", *est.coef_names_] if n in frame.index]

    lo_label = f"[{100 * alpha / 2:g}%"
    hi_label = f"{100 * (1 - alpha / 2):g}%]"
    name_w = max(14, *(len(labels[n]) for n in order)) + 2
    add(
        f"{'':<{name_w}}{'coef':>10}{'std err':>10}{'z':>9}{'P>|z|':>9}"
        f"{lo_label:>11}{hi_label:>10}"
    )
    for name in order:
        row = frame.loc[name]
        add(
            f"{labels[name]:<{name_w}}{row.coef:>10.4g}{row.std_err:>10.4g}{row.z:>9.2f}"
            f"{row.p_value:>9.3f}{row.ci_lower:>11.4g}{row.ci_upper:>10.4g}"
        )
    add("")
    for name in order:
        add(_wrap(f"{labels[name]}: {units[name]}", indent="  ", hang="    "))
    add("")

    add("Diagnostics")
    add(_dots("Outcome model R² (out-of-fold, focal nodes)", f"{d['outcome_r2']:.3f}"))
    if "treatment_auc" in d:
        add(_dots("Treatment model AUC (out-of-fold, focal nodes)", f"{d['treatment_auc']:.3f}"))
        clipped = d.get("propensity_clipped_share")
        rng = f"[{d['propensity_min']:.3f}, {d['propensity_max']:.3f}]"
        if clipped is not None:
            rng += f", {clipped:.1%} clipped"
        add(_dots("Propensity range", rng))
    else:
        add(_dots("Treatment model R² (out-of-fold, focal nodes)", f"{d['treatment_r2']:.3f}"))
    add(
        _dots(
            "Mean degree: all nodes / focal nodes",
            f"{d['mean_degree_all']:.2f} / {d['mean_degree_focal']:.2f}",
        )
    )
    if "focal_without_neighbors_share" in d:
        add(_dots("Focal nodes without neighbors", f"{d['focal_without_neighbors_share']:.1%}"))
        add(_dots("max |corr(res_T, res_peer)|", f"{d['max_abs_corr_res_t_res_peer']:.3f}"))
    if "residual_balance_max_abs_corr" in d:
        add(
            _dots(
                "Residual balance: max |corr(res_T, feature)|",
                f"{d['residual_balance_max_abs_corr']:.3f} "
                f"(noise ≈ {d['residual_balance_noise_level']:.3f})",
            )
        )
    if est.n_rep > 1:
        spread = np.ptp(est.coef_reps_, axis=0)
        add(_dots("Spread of estimates across repetitions", ", ".join(f"{v:.3g}" for v in spread)))

    if est.warnings_:
        add("")
        add("Warnings")
        for msg in est.warnings_:
            add(_wrap(f"! {msg}", indent="  ", hang="    "))

    add("")
    add("Identifying assumptions (cannot be verified from data alone)")
    if est.exposures_:
        names = ", ".join(repr(e.name) for e in est.exposures_)
        add(_wrap(f"• Exposure map: Y depends on others' treatments only through {names}."))
        add(_wrap("• Interference stays within the exposure neighborhood (check: two_hop_test)."))
    else:
        add(_wrap("• No interference: a node's outcome does not depend on others' treatments."))
    add(
        _wrap(
            "• No unobserved confounding given own and neighbor covariates "
            "(probe: negative_control_test)."
        )
    )
    add(_wrap("• Positivity: every node could plausibly receive any treatment and exposure level."))
    return Summary("\n".join(lines))


def _dots(label: str, value: str, width: int = 50) -> str:
    return f"  {label} {'.' * max(2, width - len(label))} {value}"


def _wrap(text: str, indent: str = "  ", hang: str = "    ") -> str:
    return textwrap.fill(text, width=_WIDTH, initial_indent=indent, subsequent_indent=hang)
