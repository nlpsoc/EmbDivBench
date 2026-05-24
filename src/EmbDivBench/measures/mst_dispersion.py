from __future__ import annotations

from .._accepts_text import accepts_text
from ._types import DISTANCE_METRIC, TensorLike

### Graph-Based Diversity Measure

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform
import torch



@accepts_text
def mst_dispersion(data: TensorLike,
                   metric: DISTANCE_METRIC = "cosine"
                   ) -> float:
    """Compute diversity as the total edge weight of the minimum spanning tree.

    1) Build a complete weighted graph whose edge weights are the pairwise
       distances between datapoints.
    2) Run a minimum spanning tree (Kruskal-style via SciPy) on that graph.
    3) Return the sum of edge weights of the resulting MST.

    Higher values indicate more dispersed (more diverse) data: the MST has
    to bridge larger gaps to connect every point.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays/torch.Tensor) of
            shape (n, d). Must contain at least 2 samples.
        metric: Distance metric name or callable accepted by
            ``scipy.spatial.distance.pdist``. Defaults to ``"cosine"``.

    Returns:
        float: Sum of edge weights of the minimum spanning tree of the
        complete pairwise-distance graph.

    Raises:
        ValueError: If ``data`` is not 2D or contains fewer than 2 datapoints.
    """
    # normalize input to numpy array
    if isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected shape (n, d), got {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute mst_dispersion for fewer than 2 datapoints")

    # now we create and adjacency matrix with a specified pairwise distance metric
    # by default its cosine distance
    dist_condensed = _compute_pairwise_distances(X, metric=metric)
    # need to convert it to square form
    # we use this as our adjacency matrix of a complete graph formed by all datapoints
    dist_square = squareform(dist_condensed)

    # we use the scipy minimum spanning tree implementation
    mst = minimum_spanning_tree(dist_square)

    # to obtain the mst dispersion
    # we sum the lengths of the edges required to connect all samples with the minimum total cost
    mst_dispersion = mst.sum()

    return float(mst_dispersion)
