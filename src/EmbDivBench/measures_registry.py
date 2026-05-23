"""Central measure registry used by the CLI and convenience functions."""

from __future__ import annotations

from ._registry import Registry

# Distance-based
from .measures.mean_pw_dist import mean_pw_dist
from .measures.dist_dispersion import dist_dispersion
from .measures.energy import energy
from .measures.chamfer_dist import chamfer_dist

# Geometry-based
from .measures.convex_hull_volume_2d import convex_hull_volume_2d
from .measures.span_centroid import span_centroid
from .measures.radius import radius
from .measures.diameter import diameter
from .measures.bottleneck import bottleneck
from .measures.span_medoid import span_medoid
from .measures.sum_diameter import sum_diameter
from .measures.sum_bottleneck import sum_bottleneck
from .measures.cluster_inertia import cluster_inertia

# Graph-based
from .measures.graph_entropy import graph_entropy
from .measures.mst_dispersion import mst_dispersion
from .measures.hamdiv import hamdiv

# Distribution-based
from .measures.vendi_score import vendi_score
from .measures.dcscore import dcscore
from .measures.renyi_entropy import renyi_entropy
from .measures.log_determinant import log_determinant
from .measures.bins_entropy import bins_entropy
# mag_areas is conceptually distribution-based but has a multi-dataset API,
# so it lives in the package namespace but is not registered here.

# 21 single-dataset measures, registered by category
measures = Registry()

# Distance-based
measures.register("mean_pw_dist", mean_pw_dist)
measures.register("dist_dispersion", dist_dispersion)
measures.register("energy", energy)
measures.register("chamfer_dist", chamfer_dist)

# Geometry-based
measures.register("convex_hull_volume_2d", convex_hull_volume_2d)
measures.register("span_centroid", span_centroid)
measures.register("radius", radius)
measures.register("diameter", diameter)
measures.register("bottleneck", bottleneck)
measures.register("span_medoid", span_medoid)
measures.register("sum_diameter", sum_diameter)
measures.register("sum_bottleneck", sum_bottleneck)
measures.register("cluster_inertia", cluster_inertia)

# Graph-based
measures.register("graph_entropy", graph_entropy)
measures.register("mst_dispersion", mst_dispersion)
measures.register("hamdiv", hamdiv)

# Distribution-based
measures.register("vendi_score", vendi_score)
measures.register("dcscore", dcscore)
measures.register("renyi_entropy", renyi_entropy)
measures.register("log_determinant", log_determinant)
measures.register("bins_entropy", bins_entropy)

# The single default measure
DEFAULT_MEASURE = "log_determinant"

# Curated representative set across categories
CORE_MEASURES: list[str] = [
    "log_determinant",
    "mean_pw_dist",
    "vendi_score",
    "convex_hull_volume_2d",
    "graph_entropy",
    "energy",
]
