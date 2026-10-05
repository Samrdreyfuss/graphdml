"""GraphDML: double machine learning for direct and peer effects on a network."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state

from graphdml.data import GraphData
from graphdml.exceptions import GraphDMLWarning
from graphdml.exposure import resolve_exposures
from graphdml.features import NeighborhoodFeatures, resolve_featurizer
from graphdml.focal import dependency_matrix, is_independent, select_focal_set
from graphdml.inference import aggregate_repetitions, normal_ci, sandwich_cov, solve_moment
from graphdml.learners import default_classifier, default_regressor

__all__ = ["GraphDML"]

_MAX_SEED = np.iinfo(np.int32).max
_MIN_TRAIN = 20

#: Settings that reproduce the procedure of Khatami et al. (2025) and its reference code.
PAPER_SETTINGS: dict[str, Any] = {
    "focal_set": "random",
    "nuisance_training": "focal",
    "aggregation": "dml1",
    "n_folds": 3,
    "score": "partialling-out",
}

# Thresholds for fit-time warnings.
SMALL_FOCAL_SET = 250
CLIPPED_SHARE = 0.05
COLLINEAR_CORR = 0.95
ISOLATED_SHARE = 0.5


@dataclass
class _Rep:
    focal: np.ndarray
    fold: np.ndarray
    coef: np.ndarray
    cov: np.ndarray
    R: np.ndarray
    D: np.ndarray
    y: np.ndarray
    t_hat: np.ndarray
    y_hat: np.ndarray
    res_y: np.ndarray


class GraphDML(BaseEstimator):
    """Direct and peer effects on a network via graph double machine learning.

    Estimates ``theta`` (average direct effect) and ``alpha`` (average peer effect) in the
    partially linear network model of Khatami et al. (AISTATS 2025)::

        T = m(X, A) + e_T
        Y = theta * T + alpha * (E @ T) + g(X, A) + e_Y

    where ``E`` is a known exposure map and ``m``, ``g`` are unknown functions of a node's
    own and its neighbors' covariates, learned with any scikit-learn model.

    Parameters
    ----------
    model_y : scikit-learn regressor, optional
        Learns ``E[Y | X, A]``. Defaults to :func:`graphdml.learners.default_regressor`
        (regularised histogram gradient boosting).
    model_t : scikit-learn regressor or classifier, optional
        Learns ``E[T | X, A]``. For binary ``T`` a classifier's ``predict_proba`` is used.
        Defaults to :func:`graphdml.learners.default_classifier` for binary T, else the
        regressor.
    model_g : scikit-learn regressor, optional
        Learns ``g(X, A)`` for ``score="iv-type"``. Defaults to ``model_y``.
    featurizer : object with ``transform(data)``, "own" or "neighborhood", optional
        Builds the tabular features the nuisance models see. Defaults to
        :class:`~graphdml.NeighborhoodFeatures` (own + 1- and 2-hop neighbor aggregates).
    exposure : str, ExposureMap, list or None, default="sum"
        How neighbors' treatments reach a node: "sum", "mean", "weighted", ... A list fits
        one peer coefficient per exposure. ``None`` fits the direct effect only.
    score : {"iv-type", "partialling-out"}, default="iv-type"
        "iv-type" stays consistent if either the treatment model or the outcome-confounding
        model ``g`` is right, and had lower bias in 7 of 8 benchmark cells.
        "partialling-out" is the paper's score; it is attenuated toward zero when the
        treatment model is wrong (see ``docs/methodology.md``).
    focal_set : {"min_degree", "random"}, array-like or None, default="min_degree"
        How to choose the focal set of nodes with independent scores. An array gives the
        focal nodes explicitly (validated). ``None`` uses every node and treats them as
        independent; intervals are then not valid under interference.
    nuisance_training : {"buffered", "focal"}, default="buffered"
        Which nodes train the nuisance models for each fold. "focal" (the paper) trains on
        the other folds' focal nodes only. "buffered" trains on every node whose data is
        independent of the held-out fold's scores, which is typically several times more
        data.
    n_folds : int, default=5
    n_rep : int, default=1
        Repetitions of cross-fitting (with new focal sets and folds). Estimates are
        aggregated by the median, with the variance adjusted for split-to-split spread.
    aggregation : {"dml2", "dml1"}, default="dml2"
        "dml2" solves the pooled moment condition; "dml1" (the paper) averages per-fold
        estimates.
    clip_propensity : float, default=0.01
        Clip classifier propensities to ``[c, 1 - c]``.
    mode : {None, "paper"}, default=None
        ``"paper"`` overrides the settings above to reproduce the published procedure.
    estimation_nodes : array of node positions or boolean mask, optional
        Restrict the estimate to a sub-population: only these nodes can be focal. All nodes
        still supply neighbors' treatments and nuisance training data.
    random_state : int, RandomState or None

    Attributes
    ----------
    coef_, se_, cov_ : estimates, standard errors and covariance, ordered as
        ``coef_names_`` (``"direct"`` then ``"peer:<exposure>"``).
    ade_, ape_ : the direct and (first) peer effect.
    focal_nodes_, fold_assignment_ : focal node positions and their folds (first rep).
    n_focal_ : effective sample size.
    residuals_ : per-focal-node predictions and residuals (first rep).
    diagnostics_ : dict of diagnostics; warnings_ : list of warning messages.
    """

    def __init__(
        self,
        model_y: Any = None,
        model_t: Any = None,
        *,
        model_g: Any = None,
        featurizer: Any = None,
        exposure: Any = "sum",
        score: str = "iv-type",
        focal_set: Any = "min_degree",
        nuisance_training: str = "buffered",
        n_folds: int = 5,
        n_rep: int = 1,
        aggregation: str = "dml2",
        clip_propensity: float = 0.01,
        mode: str | None = None,
        estimation_nodes: Any = None,
        random_state: Any = None,
    ) -> None:
        self.model_y = model_y
        self.model_t = model_t
        self.model_g = model_g
        self.featurizer = featurizer
        self.exposure = exposure
        self.score = score
        self.focal_set = focal_set
        self.nuisance_training = nuisance_training
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.aggregation = aggregation
        self.clip_propensity = clip_propensity
        self.mode = mode
        self.estimation_nodes = estimation_nodes
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, data: GraphData) -> GraphDML:
        """Estimate direct and peer effects from ``data``."""
        if not isinstance(data, GraphData):
            raise TypeError("fit expects a graphdml.GraphData instance.")
        s = self._settings()
        rng = check_random_state(self.random_state)
        n = data.n_nodes

        exposures = resolve_exposures(self.exposure)
        E = [sp.csr_array(e.matrix(data)) for e in exposures]
        featurizer = resolve_featurizer(s["featurizer"])
        F = np.asarray(featurizer.transform(data), dtype=float)
        if F.ndim != 2 or F.shape[0] != n:
            raise ValueError(f"featurizer returned shape {F.shape}; expected ({n}, k).")

        binary = data.treatment_is_binary
        model_y = self.model_y if self.model_y is not None else default_regressor()
        model_t = self.model_t
        if model_t is None:
            model_t = default_classifier() if binary else clone(model_y)
        model_g = self.model_g if self.model_g is not None else model_y
        t_is_clf = is_classifier(model_t)
        if t_is_clf and not binary:
            raise ValueError("model_t is a classifier but T is not binary 0/1; use a regressor.")

        dep = dependency_matrix(n, E)
        Z = np.column_stack([Ek @ data.T for Ek in E]) if E else np.empty((n, 0))

        reps = [
            self._fit_once(
                data, F, E, Z, dep, model_y, model_t, model_g, t_is_clf, s,
                np.random.RandomState(rng.randint(_MAX_SEED)),
            )
            for _ in range(self.n_rep)
        ]
        coefs = np.array([r.coef for r in reps])
        covs = np.array([r.cov for r in reps])
        if len(reps) == 1:
            coef, cov = coefs[0], covs[0]
        else:
            coef, cov = aggregate_repetitions(coefs, covs)

        rep = reps[0]
        self._final_stage = (rep.R, rep.D, rep.y)
        self.coef_ = coef
        self.cov_ = cov
        self.se_ = np.sqrt(np.diag(cov))
        self.coef_names_ = ["direct"] + [f"peer:{e.name}" for e in exposures]
        self.coef_reps_ = coefs
        self.se_reps_ = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
        self.exposures_ = exposures
        self.settings_ = {**s, "featurizer": featurizer}
        self.treatment_is_binary_ = binary
        self.n_nodes_ = n
        self.n_focal_ = len(rep.focal)
        self.focal_nodes_ = rep.focal
        self.fold_assignment_ = rep.fold
        self.residuals_ = self._residual_frame(data, rep)
        self.diagnostics_ = self._diagnostics(data, F, E, rep, t_is_clf, s)
        self.warnings_ = self._warnings(s, E)
        for msg in self.warnings_:
            warnings.warn(msg, GraphDMLWarning, stacklevel=2)
        return self

    def _fit_once(self, data, F, E, Z, dep, model_y, model_t, model_g, t_is_clf, s, rng) -> _Rep:
        n = data.n_nodes
        T, Y = data.T, data.Y
        iid = s["focal_set"] is None
        focal = self._focal_nodes(dep, s["focal_set"], rng, n)
        n_f, K = len(focal), s["n_folds"]
        if n_f < 2 * K:
            raise ValueError(f"Only {n_f} focal nodes; need at least {2 * K} for {K} folds.")

        fold = np.empty(n_f, dtype=np.int64)
        kf = KFold(K, shuffle=True, random_state=rng.randint(_MAX_SEED))
        for k, (_, test) in enumerate(kf.split(focal)):
            fold[test] = k

        res_t = np.empty(n_f)
        res_z = np.empty((n_f, len(E)))
        res_y = np.empty(n_f)
        t_hat = np.empty(n_f)
        y_hat = np.empty(n_f)
        train_y_sets = []
        for k in range(K):
            pos = np.flatnonzero(fold == k)
            est = focal[pos]
            # Nodes whose noise enters the held-out scores (the union of dependency sets).
            in_U = np.asarray(dep[est].sum(axis=0)).ravel() > 0
            U = np.flatnonzero(in_U)
            if iid or s["nuisance_training"] == "focal":
                train_t = train_y = focal[fold != k]
            else:
                train_t = np.flatnonzero(~in_U)
                train_y = np.flatnonzero((dep @ in_U.astype(float)) == 0)
            for name, idx in (("treatment", train_t), ("outcome", train_y)):
                if len(idx) < _MIN_TRAIN:
                    raise ValueError(
                        f"Fold {k}: only {len(idx)} nodes available to train the {name} model. "
                        "Use more folds or nuisance_training='focal'."
                    )

            mt = _fit_clone(model_t, F[train_t], T[train_t], rng)
            resid = np.zeros(n)
            resid[U] = T[U] - self._predict_t(mt, F[U], t_is_clf)
            res_t[pos] = resid[est]
            t_hat[pos] = T[est] - resid[est]
            for e, Ek in enumerate(E):
                res_z[pos, e] = Ek[est] @ resid

            my = _fit_clone(model_y, F[train_y], Y[train_y], rng)
            y_hat[pos] = my.predict(F[est])
            res_y[pos] = Y[est] - y_hat[pos]
            train_y_sets.append(train_y)

        R = np.column_stack([res_t, res_z])
        if s["score"] == "partialling-out":
            D, y = R, res_y
        else:
            D_all = np.column_stack([T, Z])
            preliminary = solve_moment(R, R, res_y)
            g_hat = np.empty(n_f)
            for k in range(K):
                pos = np.flatnonzero(fold == k)
                tr = train_y_sets[k]
                mg = _fit_clone(model_g, F[tr], Y[tr] - D_all[tr] @ preliminary, rng)
                g_hat[pos] = mg.predict(F[focal[pos]])
            D, y = D_all[focal], Y[focal] - g_hat

        if s["aggregation"] == "dml2":
            coef = solve_moment(R, D, y)
        else:
            coef = np.mean(
                [solve_moment(R[fold == k], D[fold == k], y[fold == k]) for k in range(K)],
                axis=0,
            )
        cov = sandwich_cov(R, D, y, coef)
        return _Rep(focal, fold, coef, cov, R, D, y, t_hat, y_hat, res_y)

    def _focal_nodes(self, dep, spec, rng, n) -> np.ndarray:
        candidates = None
        if self.estimation_nodes is not None:
            candidates = np.asarray(self.estimation_nodes)
            if candidates.dtype == bool:
                if candidates.shape != (n,):
                    raise ValueError("A boolean estimation_nodes mask needs one entry per node.")
                candidates = np.flatnonzero(candidates)
            candidates = np.unique(candidates.astype(np.int64))
        if spec is None:
            return np.arange(n) if candidates is None else candidates
        if isinstance(spec, str):
            return select_focal_set(dep, spec, rng, candidates=candidates)
        focal = np.unique(np.asarray(spec, dtype=np.int64))
        if focal.size == 0 or focal.min() < 0 or focal.max() >= n:
            raise ValueError("focal_set indices must be node positions in [0, n_nodes).")
        if not is_independent(dep, focal):
            raise ValueError(
                "The given focal nodes have overlapping dependency sets, so their scores are "
                "not independent. Use graphdml.focal.select_focal_set to build a valid set."
            )
        return focal

    def _predict_t(self, model, F, t_is_clf) -> np.ndarray:
        if not t_is_clf:
            return model.predict(F)
        classes = list(model.classes_)
        if 1 not in classes and 1.0 not in classes:
            raise ValueError("A treatment-model training fold contains no treated nodes.")
        p = model.predict_proba(F)[:, classes.index(1)]
        c = self.clip_propensity
        return np.clip(p, c, 1 - c) if c else p

    # ------------------------------------------------------------------ settings
    def _settings(self) -> dict[str, Any]:
        s: dict[str, Any] = {
            "score": self.score,
            "focal_set": self.focal_set,
            "nuisance_training": self.nuisance_training,
            "n_folds": self.n_folds,
            "aggregation": self.aggregation,
            "featurizer": self.featurizer,
        }
        if self.mode == "paper":
            s.update(PAPER_SETTINGS)
            if self.featurizer is None:
                # The paper's nuisance model is a one-layer GIN: own + summed neighbor features.
                s["featurizer"] = NeighborhoodFeatures(aggs=("sum",), hops=1, include_degree=False)
        elif self.mode is not None:
            raise ValueError(f"mode must be None or 'paper', got {self.mode!r}.")
        if s["score"] not in ("partialling-out", "iv-type"):
            raise ValueError("score must be 'partialling-out' or 'iv-type'.")
        if s["nuisance_training"] not in ("buffered", "focal"):
            raise ValueError("nuisance_training must be 'buffered' or 'focal'.")
        if s["aggregation"] not in ("dml1", "dml2"):
            raise ValueError("aggregation must be 'dml1' or 'dml2'.")
        if int(s["n_folds"]) < 2:
            raise ValueError("n_folds must be >= 2.")
        if int(self.n_rep) < 1:
            raise ValueError("n_rep must be >= 1.")
        if isinstance(s["focal_set"], str) and s["focal_set"] not in ("min_degree", "random"):
            raise ValueError("focal_set must be 'min_degree', 'random', None or node indices.")
        return s

    # ------------------------------------------------------------------ results
    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("This GraphDML instance is not fitted yet; call fit(data).")

    @property
    def ade_(self) -> float:
        """Average direct effect."""
        self._check_fitted()
        return float(self.coef_[0])

    @property
    def ape_(self) -> float | None:
        """Average peer effect for the first exposure map (None if no exposure)."""
        self._check_fitted()
        return float(self.coef_[1]) if len(self.coef_) > 1 else None

    def summary_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """Estimates, standard errors, z statistics, p-values and confidence intervals."""
        self._check_fitted()
        lo, hi = normal_ci(self.coef_, self.se_, alpha)
        z = self.coef_ / self.se_
        return pd.DataFrame(
            {
                "coef": self.coef_,
                "std_err": self.se_,
                "z": z,
                "p_value": 2 * stats.norm.sf(np.abs(z)),
                "ci_lower": lo,
                "ci_upper": hi,
            },
            index=pd.Index(self.coef_names_, name="effect"),
        )

    def cluster_summary_frame(self, groups: Any, alpha: float = 0.05) -> pd.DataFrame:
        """Like :meth:`summary_frame`, with standard errors clustered by ``groups``.

        ``groups`` gives a cluster label for every node (e.g. village). Use it when nodes
        may share common shocks beyond their dependency sets. Requires ``n_rep=1``.
        """
        self._check_fitted()
        if self.n_rep != 1:
            raise ValueError("cluster_summary_frame requires n_rep=1.")
        R, D, y = self._final_stage
        g = np.asarray(groups)[self.focal_nodes_]
        se = np.sqrt(np.diag(sandwich_cov(R, D, y, self.coef_, groups=g)))
        lo, hi = normal_ci(self.coef_, se, alpha)
        z = self.coef_ / se
        return pd.DataFrame(
            {"coef": self.coef_, "std_err": se, "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)),
             "ci_lower": lo, "ci_upper": hi},
            index=pd.Index(self.coef_names_, name="effect"),
        )

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        return self.summary_frame(alpha)[["ci_lower", "ci_upper"]]

    def summary(self, alpha: float = 0.05):
        """Human-readable results, diagnostics, warnings and identifying assumptions."""
        from graphdml.results import format_summary

        self._check_fitted()
        return format_summary(self, alpha)

    def _residual_frame(self, data: GraphData, rep: _Rep) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "node_id": data.node_ids[rep.focal],
                "node_index": rep.focal,
                "fold": rep.fold,
                "t_hat": rep.t_hat,
                "y_hat": rep.y_hat,
                "res_t": rep.R[:, 0],
            }
        )
        for j, name in enumerate(self.coef_names_[1:], start=1):
            frame[f"res_{name}"] = rep.R[:, j]
        frame["res_y"] = rep.res_y
        return frame

    def _diagnostics(self, data, F, E, rep: _Rep, t_is_clf, s) -> dict[str, float]:
        focal = rep.focal
        n_f = len(focal)
        T_f, Y_f = data.T[focal], data.Y[focal]
        res_t = rep.R[:, 0]
        d: dict[str, float] = {
            "n_nodes": data.n_nodes,
            "n_focal": n_f,
            "focal_share": n_f / data.n_nodes,
            "mean_degree_all": float(data.degree.mean()),
            "mean_degree_focal": float(data.degree[focal].mean()),
            "outcome_r2": _r2(Y_f, rep.res_y),
        }
        if data.treatment_is_binary:
            both = 0 < T_f.sum() < n_f
            d["treatment_auc"] = float(roc_auc_score(T_f, rep.t_hat)) if both else np.nan
            d["propensity_min"] = float(rep.t_hat.min())
            d["propensity_max"] = float(rep.t_hat.max())
            c = self.clip_propensity
            if t_is_clf and c:
                at_bound = (rep.t_hat <= c + 1e-12) | (rep.t_hat >= 1 - c - 1e-12)
                d["propensity_clipped_share"] = float(at_bound.mean())
        else:
            d["treatment_r2"] = _r2(T_f, res_t)
        if E:
            has_nbrs = np.diff(E[0].indptr)[focal] > 0
            d["focal_without_neighbors_share"] = float(1 - has_nbrs.mean())
            corrs = [_corr(res_t, rep.R[:, j]) for j in range(1, rep.R.shape[1])]
            d["max_abs_corr_res_t_res_peer"] = float(np.nanmax(np.abs(corrs)))
        J = rep.R.T @ rep.D / n_f
        d["jacobian_condition_number"] = float(np.linalg.cond(J))
        Ff = F[focal]
        keep = Ff.std(axis=0) > 0
        if keep.any():
            balance = [_corr(res_t, Ff[:, j]) for j in np.flatnonzero(keep)]
            d["residual_balance_max_abs_corr"] = float(np.nanmax(np.abs(balance)))
            d["residual_balance_noise_level"] = float(3 / np.sqrt(n_f))
        return d

    def _warnings(self, s, E) -> list[str]:
        d = self.diagnostics_
        out = []
        if s["focal_set"] is None and E:
            out.append(
                "focal_set=None treats all nodes as independent; confidence intervals are not "
                "valid when neighbors' scores are dependent."
            )
        if d["n_focal"] < SMALL_FOCAL_SET:
            out.append(
                f"Only {d['n_focal']} focal nodes (the effective sample size). Expect wide "
                "intervals and a less reliable normal approximation."
            )
        if d.get("propensity_clipped_share", 0) > CLIPPED_SHARE:
            out.append(
                f"{d['propensity_clipped_share']:.0%} of focal propensities were clipped: weak "
                "overlap between treated and untreated nodes."
            )
        if d.get("max_abs_corr_res_t_res_peer", 0) > COLLINEAR_CORR:
            out.append(
                "Direct and peer residuals are nearly collinear; the two effects are poorly "
                "separated in this graph."
            )
        if d.get("focal_without_neighbors_share", 0) > ISOLATED_SHARE:
            out.append(
                f"{d['focal_without_neighbors_share']:.0%} of focal nodes have no neighbors, "
                "so the peer effect is identified from few nodes."
            )
        return out


def _fit_clone(model, X, y, rng):
    est = clone(model)
    params = est.get_params(deep=True)
    seeds = {
        k: rng.randint(_MAX_SEED)
        for k, v in params.items()
        if (k == "random_state" or k.endswith("__random_state")) and v is None
    }
    if seeds:
        est.set_params(**seeds)
    return est.fit(X, y)


def _r2(y: np.ndarray, resid: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum(resid**2) / ss) if ss > 0 else np.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
