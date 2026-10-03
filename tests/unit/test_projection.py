"""Projection tests: the frame is linear (queries land with the corpus), the fit
reports honest variance, and degenerate inputs fail loudly."""

from __future__ import annotations

import numpy as np
import pytest

from reglens.indexing.projection import fit_pca


def clustered_vectors(seed: int = 7) -> np.ndarray:
    """Three tight clusters in 8-D, so the first axes are well defined."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 8)) * 4.0
    return np.vstack(
        [center + rng.normal(scale=0.3, size=8) for center in centers for _ in range(20)]
    )


def test_fit_reports_shapes_and_descending_variance() -> None:
    basis = fit_pca(clustered_vectors())
    assert basis.components.shape == (3, 8)
    assert basis.mean.shape == (8,)
    assert len(basis.explained_variance_ratio) == 3
    assert all(0.0 <= value <= 1.0 for value in basis.explained_variance_ratio)
    assert list(basis.explained_variance_ratio) == sorted(
        basis.explained_variance_ratio, reverse=True
    )
    # Axis rows are unit length (a scale change would silently distort distances).
    norms = np.linalg.norm(basis.components, axis=1)
    assert np.allclose(norms, 1.0)


def test_project_is_linear_and_consistent_with_the_fit() -> None:
    vectors = clustered_vectors()
    basis = fit_pca(vectors)
    coords = basis.project(vectors)
    assert coords.shape == (len(vectors), 3)

    # A new vector (the live query) lands in the same frame: projecting the whole
    # corpus and one row separately must agree.
    single = basis.project(vectors[:1])
    assert np.allclose(single[0], coords[0])

    # Centres stay put under linearity: mean of projected == projected mean.
    assert np.allclose(coords.mean(axis=0), basis.project(vectors.mean(axis=0, keepdims=True))[0])


def test_clusters_are_separated_on_the_first_axes() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0.0, scale=0.2, size=(30, 8))
    b = rng.normal(loc=6.0, scale=0.2, size=(30, 8))
    vectors = np.vstack([a, b])
    coords = fit_pca(vectors).project(vectors)
    # The dominant variance is the a-vs-b split: the group centres must be far apart
    # relative to each group's own spread (measured from the group mean, not the
    # global one — that sits between the clusters by construction).
    gap = np.linalg.norm(coords[:30].mean(axis=0) - coords[30:].mean(axis=0))
    spread_a = np.linalg.norm(coords[:30] - coords[:30].mean(axis=0), axis=1).mean()
    spread_b = np.linalg.norm(coords[30:] - coords[30:].mean(axis=0), axis=1).mean()
    assert gap > 4 * (spread_a + spread_b) / 2


def test_degenerate_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_pca(np.zeros(4))
    with pytest.raises(ValueError, match="more than 3 vectors"):
        fit_pca(np.zeros((3, 8)))
    with pytest.raises(ValueError, match="fewer than 3 components"):
        fit_pca(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="NaN or Infinity"):
        fit_pca(np.full((10, 8), np.nan))
    with pytest.raises(ValueError, match="zero variance"):
        fit_pca(np.ones((10, 8)))
