from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.aggregation import (
    AveragingMaskBuilderMixin,
    MaxMaskBuilderMixin,
)
from ratiopath.masks.mask_builders.factory import (
    MaskBuilderFactory,
    ScalarMeanMB,
    ScalarMaxMB,
    MemMapMeanMB,
    MemMapMaxMB,
    ExplicitCoordMemMapMeanMB,
    ExplicitCoordMemMapMaxMB,
    MeanMB,
    MaxMB,
    ExplicitCoordMeanMB,
    ExplicitCoordMaxMB,
    ExplicitCoordScalarMeanMB,
    ExplicitCoordScalarMaxMB,
)
from ratiopath.masks.mask_builders.mask_builder import (
    AccumulatorType,
)
from ratiopath.masks.mask_builders.receptive_field_manipulation import (
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    ScalarUniformTiledMaskBuilder,
)
from ratiopath.masks.mask_builders.storage import (
    NumpyArrayMaskBuilderAllocatorMixin,
    NumpyMemMapMaskBuilderAllocatorMixin,
)


__all__ = [
    "ScalarMeanMB",
    "ScalarMaxMB",
    "MemMapMeanMB",
    "MemMapMaxMB",
    "ExplicitCoordMemMapMeanMB",
    "ExplicitCoordMemMapMaxMB",
    "MeanMB",
    "MaxMB",
    "ExplicitCoordMeanMB",
    "ExplicitCoordMaxMB",
    "ExplicitCoordScalarMeanMB",
    "ExplicitCoordScalarMaxMB",
    "MaskBuilderFactory",
]

# ============================================================================
# NOTE: All concrete mask builder classes are now defined in factory.py
# and imported above. The old class definitions that were previously in this
# file have been removed. See factory.py for the implementation details.
# ============================================================================
