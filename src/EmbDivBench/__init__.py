"""EmbDivBench – embedding-based diversity measures for text and vector data."""

from __future__ import annotations

### Distance-Based Diversity Measures
from .measures.mean_pw_dist import mean_pw_dist
from .measures.dist_dispersion import dist_dispersion
from .measures.energy import energy
from .measures.chamfer_dist import chamfer_dist

### Geometry-Based Diversity Measures
from .measures.convex_hull_volume_2d import convex_hull_volume_2d
from .measures.span_centroid import span_centroid
from .measures.radius import radius
from .measures.diameter import diameter
from .measures.bottleneck import bottleneck
from .measures.span_medoid import span_medoid
from .measures.sum_diameter import sum_diameter
from .measures.sum_bottleneck import sum_bottleneck
from .measures.cluster_inertia import cluster_inertia

### Graph-Based Diversity Measures
from .measures.graph_entropy import graph_entropy
from .measures.mst_dispersion import mst_dispersion
from .measures.hamdiv import hamdiv

### Distribution-Based Diversity Measures
from .measures.vendi_score import vendi_score
from .measures.dcscore import dcscore
from .measures.renyi_entropy import renyi_entropy
from .measures.log_determinant import log_determinant
from .measures.bins_entropy import bins_entropy
from .measures.mag_areas import mag_areas

### Registries
from .axes_registry import axes
from .measures_registry import measures

### Embedding helper
from .embed import embed_texts

### Main entry point
from .convenience import measure_diversity

### Caching utilities
from .compute_pairwise import compute_pairwise_distances, clear_distance_cache, distance_cache_info


__all__ = [
    # Main entry point
    "measure_diversity",
    # Individual measures
    "mean_pw_dist", "dist_dispersion", "energy", "chamfer_dist",
    "convex_hull_volume_2d", "span_centroid", "radius", "diameter", "bottleneck",
    "span_medoid", "sum_diameter", "sum_bottleneck", "cluster_inertia",
    "graph_entropy", "mst_dispersion", "hamdiv",
    "vendi_score", "dcscore", "renyi_entropy", "log_determinant",
    "bins_entropy", "mag_areas",
    # Helpers
    "embed_texts",
    # Registries
    "axes", "measures",
    # Pairwise distance caching
    "compute_pairwise_distances", "clear_distance_cache", "distance_cache_info",
]
