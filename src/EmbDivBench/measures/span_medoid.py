from __future__ import annotations

from typing import Any, Sequence

from .._accepts_text import accepts_text
from ._types import DISTANCE_METRIC

### Geometry-Based Diversity Measure

import numpy as np
from scipy.spatial.distance import squareform

from .utils import _compute_pairwise_distances



@accepts_text
def span_medoid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any,
) -> float:
    """Compute the Span with Medoid diversity measure (Cox et al., 2021).

    1) Compute all unique pairwise distances and form the full distance matrix.
    2) Identify the medoid: the point whose total distance to every other
       point is minimal (``argmin_i sum_j d(x_i, x_j)``).
    3) Return the mean distance from every point to the medoid.

    Higher values indicate higher spread (diversity) around the medoid.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
            Must contain at least 2 samples.
        metric: Distance metric name or callable accepted by
            ``scipy.spatial.distance.pdist``. Defaults to ``"cosine"``.
        **metric_kwargs: Extra keyword arguments forwarded to ``pdist``.

    Returns:
        float: Mean distance from all points to the medoid.

    Raises:
        ValueError:
            If data is empty or contains fewer than 2 datapoints.
    """
    # 1) pairwise distances (condensed) -> full matrix (n, n)
    dist_vec = _compute_pairwise_distances(data, metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)  # symmetric, zeros on diagonal

    # sum of distances for each row
    row_sums = dist_mat.sum(axis=1)

    # 3) medoid = the row with the minimum sum of distances
    medoid_idx = int(np.argmin(row_sums))

    # 4) distances to the medoid, and take the average
    dists_to_medoid = dist_mat[:, medoid_idx]
    return float(np.mean(dists_to_medoid))
