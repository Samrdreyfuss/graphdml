"""The :class:`GraphDataset` bundle returned by every dataset loader and simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from graphdml.data import GraphData

__all__ = ["GraphDataset"]


@dataclass(repr=False)
class GraphDataset:
    """A ready-to-fit :class:`~graphdml.GraphData` plus its ground truth and story.

    Attributes
    ----------
    name : str
    data : GraphData
    truth : dict
        ``{"ade": ..., "ape": ...}``: the true average direct and peer effects.
    exposure : str
        The exposure map the data were generated with.
    treatment_name, outcome_name : str
        What ``T`` and ``Y`` mean in the story.
    DESCR : str
        The story card: what the variables are, what confounds, what naive methods miss.
    node_effects : dict of arrays
        Per-node effects (``"ade"``, ``"ape"``) when effects vary across nodes.
    extras : dict
        Anything else: oracle nuisances, negative-control outcomes, positions, latents.
    """

    name: str
    data: GraphData
    truth: dict[str, float]
    exposure: str
    treatment_name: str = "T"
    outcome_name: str = "Y"
    DESCR: str = ""
    node_effects: dict[str, np.ndarray] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def nodes_df(self) -> pd.DataFrame:
        nodes, _ = self.data.to_pandas()
        return nodes.rename(columns={"T": self.treatment_name, "Y": self.outcome_name})

    @property
    def edges_df(self) -> pd.DataFrame:
        return self.data.to_pandas()[1]

    def focal_truth(self, nodes: Any) -> dict[str, float]:
        """Average true effects over a subset of nodes (e.g. ``model.focal_nodes_``).

        Equals :attr:`truth` when effects are the same for every node.
        """
        nodes = np.asarray(nodes)
        return {
            key: float(self.node_effects[key][nodes].mean()) if key in self.node_effects else v
            for key, v in self.truth.items()
        }

    def __repr__(self) -> str:
        truth = ", ".join(f"{k.upper()}={v:g}" for k, v in self.truth.items())
        return (
            f"GraphDataset({self.name!r}: {self.data.n_nodes:,} nodes, "
            f"{self.data.n_edges:,} edges, T={self.treatment_name!r}, "
            f"Y={self.outcome_name!r}, exposure={self.exposure!r}, truth: {truth})"
        )


def load_descr(name: str) -> str:
    try:
        return resources.files("graphdml.datasets").joinpath(f"descr/{name}.md").read_text()
    except FileNotFoundError:
        return ""
