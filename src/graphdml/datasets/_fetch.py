"""Real networks, downloaded on demand and cached (never redistributed with graphdml)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graphdml.simulate.graphs import edges_to_adjacency

__all__ = ["RealNetwork", "fetch_lastfm_asia", "get_data_home"]

_LASTFM_URL = "https://snap.stanford.edu/data/lastfm_asia.zip"
_LASTFM_SHA256 = "51acb78a923bb223ed6e61be88f91122fb29adca3f07beff7289cafd98601d47"
_LASTFM_CITATION = (
    "Rozemberczki, B. and Sarkar, R. (2020). Characteristic Functions on Graphs: Birds of a "
    "Feather, from Statistical Descriptors to Parametric Models. CIKM 2020. Data: SNAP, "
    "https://snap.stanford.edu/data/feather-lastfm-social.html"
)


@dataclass(repr=False)
class RealNetwork:
    """A real network with its node attributes, before any treatment or outcome."""

    name: str
    adjacency: sp.csr_array
    attributes: dict[str, Any]
    citation: str

    def __repr__(self) -> str:
        n = self.adjacency.shape[0]
        return (
            f"RealNetwork({self.name!r}: {n:,} nodes, {self.adjacency.nnz // 2:,} edges, "
            f"attributes={sorted(self.attributes)})"
        )


def get_data_home(data_home: str | os.PathLike | None = None) -> Path:
    """Cache directory: ``data_home``, ``$GRAPHDML_DATA``, or ``~/.cache/graphdml``."""
    path = Path(data_home or os.environ.get("GRAPHDML_DATA", "~/.cache/graphdml")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_lastfm_asia(
    data_home: str | os.PathLike | None = None, download: bool = True
) -> RealNetwork:
    """LastFM Asia social network (SNAP): 7,624 users, 27,806 mutual-follow edges.

    Attributes: ``likes`` (sparse users × 7,842 artists, 1 = liked) and ``country``
    (anonymised country code). Downloaded from SNAP on first use (~6.5 MB), checksum
    verified, then cached. Cite ``RealNetwork.citation`` when you use it.
    """
    path = get_data_home(data_home) / "lastfm_asia.zip"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False.")
        with urllib.request.urlopen(_LASTFM_URL, timeout=60) as r:
            payload = r.read()
        if hashlib.sha256(payload).hexdigest() != _LASTFM_SHA256:
            raise OSError("Checksum mismatch for lastfm_asia.zip; refusing to use it.")
        path.write_bytes(payload)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != _LASTFM_SHA256:
        raise OSError(f"Checksum mismatch for cached {path}; delete it and retry.")

    with zipfile.ZipFile(path) as z:
        folder = next(n for n in z.namelist() if n.endswith("/")).rstrip("/")
        edges = pd.read_csv(io.BytesIO(z.read(f"{folder}/lastfm_asia_edges.csv")))
        target = pd.read_csv(io.BytesIO(z.read(f"{folder}/lastfm_asia_target.csv")))
        features = json.loads(z.read(f"{folder}/lastfm_asia_features.json"))

    n = len(features)
    rows = np.concatenate([np.full(len(v), int(k)) for k, v in features.items()])
    cols = np.concatenate([np.asarray(v, dtype=int) for v in features.values()])
    likes = sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, int(cols.max()) + 1))
    likes.sum_duplicates()
    likes.data[:] = 1.0
    country = target.sort_values("id")["target"].to_numpy()
    A = edges_to_adjacency(n, edges.to_numpy())
    return RealNetwork("lastfm_asia", A, {"likes": likes, "country": country}, _LASTFM_CITATION)
