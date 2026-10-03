"""PCA projection of stored embeddings into a viewable 3D frame.

Why PCA rather than a fancier embedding-space viewer:

* The view must accept a **new** vector — the live query — and land it in the same
  frame. PCA is a fixed linear map, so one dot product projects the query; t-SNE has
  no out-of-sample transform, and UMAP would add a dependency to do worse at this.
* It must be honest about what it throws away: the fit reports how much variance each
  axis captures, and the UI shows those numbers next to the picture.
* The ranking shown alongside the map never comes from these three coordinates.
  Retrieval decides on the full stored-dimension cosine; the projection is only the
  view. A prettier map that ranked points would be a lie about how the system works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_COMPONENTS = 3


@dataclass(frozen=True)
class ProjectionBasis:
    """A fitted linear map from the embedding space to ``N_COMPONENTS`` axes."""

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (N_COMPONENTS, d), rows are unit-length principal axes
    explained_variance_ratio: tuple[float, ...]  # per axis, descending

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """``(n, d)`` vectors → ``(n, N_COMPONENTS)`` coordinates."""
        centered = np.asarray(vectors, dtype=np.float64) - self.mean
        return centered @ self.components.T


def fit_pca(vectors: np.ndarray, *, n_components: int = N_COMPONENTS) -> ProjectionBasis:
    """Fit the top ``n_components`` principal axes by economy SVD of the centred data.

    Deterministic for a given input (LAPACK ``gesdd``); the *sign* of an axis is not
    meaningful, only the frame it defines with the others.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix of vectors, got shape {matrix.shape}")
    if matrix.shape[0] <= n_components:
        raise ValueError(
            f"need more than {n_components} vectors to fit {n_components} components, "
            f"got {matrix.shape[0]}"
        )
    if matrix.shape[1] < n_components:
        raise ValueError(
            f"vectors have {matrix.shape[1]} dimensions, fewer than {n_components} components"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain NaN or Infinity; refusing to fit")

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    total_variance = float((singular_values**2).sum())
    if total_variance == 0.0:
        raise ValueError("vectors have zero variance; there is no space to project")
    explained = tuple(
        float((singular_values[index] ** 2) / total_variance)
        for index in range(n_components)
    )
    return ProjectionBasis(mean=mean, components=components, explained_variance_ratio=explained)
