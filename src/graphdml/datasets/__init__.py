"""Example datasets with a known ground truth.

=================================  ===========================================================
``make_flu_town``                  direct vs. peer effects; network confounding
``make_referral_app``              choosing an exposure map ("sum" vs "mean"); hub-heavy graphs
``make_classroom_tutoring``        continuous treatment; communities; focal-set estimand
``make_homophily_trap``            a failure mode (latent homophily) and how to detect it
``make_toy_graph``                 a 30-node graph for pictures of focal sets and folds
``make_signal_check``              planted effects plus a null twin, for graphdml.selftest()
``make_lastfm_promo``              semi-synthetic: real LastFM Asia network and covariates
``fetch_lastfm_asia``              the raw LastFM Asia network (downloaded and cached)
``load_insurance_experiment``      real randomized network experiment (Cai et al. 2015)
=================================  ===========================================================

Each returns a :class:`GraphDataset`; read ``dataset.DESCR`` for its story.
"""

from graphdml.datasets._base import GraphDataset
from graphdml.datasets._fetch import RealNetwork, fetch_lastfm_asia
from graphdml.datasets._insurance import load_insurance_experiment
from graphdml.datasets._semisynthetic import make_lastfm_promo
from graphdml.datasets._signal import make_signal_check
from graphdml.datasets._stories import (
    make_classroom_tutoring,
    make_flu_town,
    make_homophily_trap,
    make_referral_app,
    make_toy_graph,
)

__all__ = [
    "GraphDataset",
    "RealNetwork",
    "fetch_lastfm_asia",
    "load_insurance_experiment",
    "make_classroom_tutoring",
    "make_flu_town",
    "make_homophily_trap",
    "make_lastfm_promo",
    "make_referral_app",
    "make_signal_check",
    "make_toy_graph",
]
