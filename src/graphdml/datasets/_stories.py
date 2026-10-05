"""Synthetic "story" datasets: intuitive settings with a known ground truth.

Each dataset teaches one idea (see its ``DESCR``). All are generated on the fly, so they
are tunable, need no downloads and carry no licensing constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.utils import check_random_state

from graphdml.data import GraphData
from graphdml.datasets._base import GraphDataset, load_descr
from graphdml.exposure import row_normalize
from graphdml.simulate.graphs import (
    barabasi_albert,
    edges_to_adjacency,
    geometric_graph,
    random_geometric,
    stochastic_block_model,
)

__all__ = [
    "make_classroom_tutoring",
    "make_flu_town",
    "make_homophily_trap",
    "make_referral_app",
    "make_toy_graph",
]


def make_flu_town(
    n_residents: int = 5000,
    *,
    avg_neighbors: float = 6.0,
    confounding: float = 1.0,
    direct_effect: float = -2.0,
    peer_effect: float = -0.5,
    random_state: Any = 0,
) -> GraphDataset:
    """Flu shots in a town: direct protection, herd protection and network confounding.

    Treatment ``flu_shot`` (binary), outcome ``sick_days`` this season, exposure ``"sum"``
    (number of vaccinated neighbors). Health-conscious residents cluster together, and
    neighbors' health-consciousness drives both vaccination and sickness. Naive regression
    overstates how much the shot protects you. See ``DESCR`` for the full story.
    """
    rng = check_random_state(random_state)
    A, pos = random_geometric(n_residents, avg_neighbors, rng, return_positions=True)
    M = row_normalize(A)
    n = n_residents

    age = rng.uniform(18, 85, n)
    chronic = rng.binomial(1, expit(-2.0 + 0.05 * (age - 50))).astype(float)
    z = rng.standard_normal(n)
    hc = z + M @ z  # health-consciousness is spatially clustered
    hc = (hc - hc.mean()) / hc.std()
    nbr_hc = M @ hc

    logit = -0.2 + 0.9 * hc + 0.02 * (age - 50) + 0.6 * chronic + confounding * nbr_hc
    T = rng.binomial(1, expit(logit)).astype(float)
    Y = (
        7.0
        + 0.05 * (age - 50)
        + 0.001 * (age - 50) ** 2
        + 2.5 * chronic
        - 0.8 * hc
        - 1.5 * confounding * nbr_hc
        + direct_effect * T
        + peer_effect * (A @ T)
        + rng.normal(0, 1.5, n)
    )
    X = np.column_stack([age, chronic, hc])
    data = GraphData(
        X, T, Y, A, feature_names=["age", "chronic_condition", "health_consciousness"]
    )
    return GraphDataset(
        name="flu_town",
        data=data,
        truth={"ade": direct_effect, "ape": peer_effect},
        exposure="sum",
        treatment_name="flu_shot",
        outcome_name="sick_days",
        DESCR=load_descr("flu_town"),
        extras={"positions": pos},
    )


def make_referral_app(
    n_users: int = 8000,
    *,
    links_per_user: int = 3,
    confounding: float = 1.0,
    direct_effect: float = 12.0,
    peer_effect: float = 40.0,
    random_state: Any = 0,
) -> GraphDataset:
    """Promo coupons in a social app: targeting on friends' engagement, mean exposure.

    Treatment ``coupon`` (binary), outcome ``monthly_spend`` in dollars, exposure
    ``"mean"`` (share of friends who got a coupon): the peer effect of 40 means +$4 for
    every additional 10% of friends with a coupon. The friendship graph has influencer hubs
    (preferential attachment), and the targeting model favours users with engaged friends.
    """
    rng = check_random_state(random_state)
    A = barabasi_albert(n_users, links_per_user, rng)
    M = row_normalize(A)
    n = n_users

    engagement = rng.normal(0, 0.6, n)
    tenure = np.round(rng.uniform(1, 60, n))
    premium = rng.binomial(1, expit(-1.5 + 1.0 * engagement)).astype(float)
    friends_engagement = M @ engagement

    logit = -0.5 + 1.2 * engagement + 1.5 * confounding * friends_engagement + 0.5 * premium
    T = rng.binomial(1, expit(logit)).astype(float)
    Y = (
        40.0
        + 12.0 * engagement
        + 4.0 * engagement**2
        + 0.3 * tenure
        + 10.0 * premium
        + 15.0 * confounding * friends_engagement
        + direct_effect * T
        + peer_effect * (M @ T)
        + rng.normal(0, 8.0, n)
    )
    X = np.column_stack([engagement, tenure, premium])
    data = GraphData(X, T, Y, A, feature_names=["engagement_score", "tenure_months", "premium"])
    return GraphDataset(
        name="referral_app",
        data=data,
        truth={"ade": direct_effect, "ape": peer_effect},
        exposure="mean",
        treatment_name="coupon",
        outcome_name="monthly_spend",
        DESCR=load_descr("referral_app"),
    )


def make_classroom_tutoring(
    n_classes: int = 200,
    class_size: int = 25,
    *,
    friends_per_student: float = 6.0,
    confounding: float = 1.0,
    direct_effect: float = 1.5,
    peer_effect: float = 0.4,
    heterogeneous: bool = False,
    random_state: Any = 0,
) -> GraphDataset:
    """Tutoring hours in classrooms: continuous treatment, communities, focal-set estimand.

    Treatment ``tutoring_hours`` (continuous), outcome ``test_score``, exposure ``"mean"``
    (friends' average tutoring hours). Friendships form mostly within classes, and students
    with higher prior GPA have more friends. With ``heterogeneous=True`` tutoring helps
    lower-GPA students more, so the focal set (which favours low-degree students) has a
    larger average effect than the whole school: compare ``truth`` with
    ``focal_truth(model.focal_nodes_)``.
    """
    rng = check_random_state(random_state)
    n = n_classes * class_size
    class_id = np.repeat(np.arange(n_classes), class_size)
    class_quality = rng.normal(0, 0.5, n_classes)[class_id]
    gpa_z = class_quality + rng.standard_normal(n)
    gpa_z = (gpa_z - gpa_z.mean()) / gpa_z.std()

    sociability = np.exp(0.4 * gpa_z)
    w = sociability / sociability.mean()
    p_in = friends_per_student / (class_size - 1)
    A = stochastic_block_model(
        [class_size] * n_classes, p_in, 0.5 / n, random_state=rng, weights=w
    )
    M = row_normalize(A)

    parent_ed = np.clip(np.round(1.5 + 0.5 * gpa_z + rng.normal(0, 0.9, n)), 0, 3)
    free_lunch = rng.binomial(1, expit(-0.8 - 0.9 * (parent_ed - 1.5))).astype(float)
    friends_parent_ed = M @ parent_ed
    friends_gpa = M @ gpa_z

    hours = np.maximum(
        0.0,
        3.0
        - 1.0 * gpa_z
        + 0.5 * (parent_ed - 1.5)
        + 0.8 * confounding * (friends_parent_ed - 1.5)
        + rng.normal(0, 1.5, n),
    )
    theta_i = direct_effect - (0.6 * gpa_z if heterogeneous else 0.0) * np.ones(n)
    Y = (
        70.0
        + 8.0 * gpa_z
        + 2.0 * (parent_ed - 1.5)
        - 3.0 * free_lunch
        + 3.0 * confounding * friends_gpa
        + theta_i * hours
        + peer_effect * (M @ hours)
        + rng.normal(0, 5.0, n)
    )
    prior_gpa = 3.0 + 0.4 * gpa_z
    X = np.column_stack([prior_gpa, parent_ed, free_lunch])
    data = GraphData(
        X, hours, Y, A, feature_names=["prior_gpa", "parent_education", "free_lunch"]
    )
    node_effects = {"ade": theta_i, "ape": np.full(n, peer_effect)} if heterogeneous else {}
    return GraphDataset(
        name="classroom_tutoring",
        data=data,
        truth={"ade": float(theta_i.mean()), "ape": peer_effect},
        exposure="mean",
        treatment_name="tutoring_hours",
        outcome_name="test_score",
        DESCR=load_descr("classroom_tutoring"),
        node_effects=node_effects,
        extras={"class_id": class_id},
    )


def make_homophily_trap(
    n_people: int = 5000,
    *,
    avg_friends: float = 6.0,
    homophily: float = 3.0,
    direct_effect: float = 1.0,
    random_state: Any = 0,
) -> GraphDataset:
    """An online course and salary growth, where the true peer effect is zero.

    Treatment ``took_course`` (binary), outcome ``salary_growth`` (%), exposure ``"mean"``.
    An unobserved trait ("ambition") drives enrolment, salary growth *and* who befriends
    whom. GraphDML reports a spurious peer effect; ``extras["negative_control"]`` (last
    year's salary growth, which the course cannot affect) exposes the problem via
    :func:`graphdml.negative_control_test`.
    """
    rng = check_random_state(random_state)
    n = n_people
    ambition = rng.standard_normal(n)
    coords = np.column_stack([homophily * ambition, rng.standard_normal(n)])
    A = geometric_graph(coords, avg_friends)

    age = rng.uniform(22, 60, n)
    education = np.clip(np.round(rng.normal(15, 2, n)), 10, 22)
    logit = -0.5 + 1.2 * ambition + 0.15 * (education - 15) - 0.02 * (age - 40)
    T = rng.binomial(1, expit(logit)).astype(float)
    base = 3.0 + 2.5 * ambition + 0.25 * (education - 15) - 0.05 * (age - 40)
    Y = base + direct_effect * T + rng.normal(0, 1.5, n)
    Y_pre = base + rng.normal(0, 1.5, n)

    X = np.column_stack([age, education])
    data = GraphData(X, T, Y, A, feature_names=["age", "education_years"])
    return GraphDataset(
        name="homophily_trap",
        data=data,
        truth={"ade": direct_effect, "ape": 0.0},
        exposure="mean",
        treatment_name="took_course",
        outcome_name="salary_growth",
        DESCR=load_descr("homophily_trap"),
        extras={"negative_control": Y_pre, "latent_ambition": ambition},
    )


def make_toy_graph(random_state: Any = 0) -> GraphDataset:
    """A hand-drawn 30-node graph for illustrating focal sets and folds (not estimation).

    Three clusters joined by short chains, plus two isolated nodes. ``extras["positions"]``
    holds a fixed 2-D layout for plotting.
    """
    rng = check_random_state(random_state)
    edges = []
    centers = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.5)]
    pos = np.zeros((30, 2))
    for c, (cx, cy) in enumerate(centers):
        base = 8 * c
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pos[base : base + 8] = np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
        edges += [(base + i, base + (i + 1) % 8) for i in range(8)]
        edges += [(base + 0, base + 4), (base + 2, base + 6)]
    edges += [(3, 24), (24, 25), (25, 12), (15, 26), (26, 27), (27, 20)]
    pos[24], pos[25] = (1.9, -0.6), (2.7, -0.6)
    pos[26], pos[27] = (4.0, 1.6), (3.2, 2.6)
    pos[28], pos[29] = (5.0, 3.5), (5.6, 3.5)
    A = edges_to_adjacency(30, np.asarray(edges))

    x = rng.standard_normal(30)
    T = rng.binomial(1, expit(x)).astype(float)
    Y = 1.0 * T + 0.5 * (A @ T) + x + rng.normal(0, 0.5, 30)
    data = GraphData(x[:, None], T, Y, A, feature_names=["x"])
    return GraphDataset(
        name="toy_graph",
        data=data,
        truth={"ade": 1.0, "ape": 0.5},
        exposure="sum",
        DESCR=load_descr("toy_graph"),
        extras={"positions": pos},
    )
