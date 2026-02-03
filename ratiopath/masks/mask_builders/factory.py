from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.aggregation import (
    AveragingMaskBuilderMixin,
    MaxMaskBuilderMixin,
)
from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilderABC
from ratiopath.masks.mask_builders.receptive_field_manipulation import (
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    ScalarUniformTiledMaskBuilder,
)
from ratiopath.masks.mask_builders.storage import (
    NumpyArrayMaskBuilderAllocatorMixin,
    NumpyMemMapMaskBuilderAllocatorMixin,
)


class MemMapAveragingMaskBuilderMixin(AveragingMaskBuilderMixin):
    """Averaging mixin with explicit memmap overlap counter allocation.

    This mixin mirrors the previous concrete memmap averaging builder behavior by
    allocating the main accumulator and overlap counter as separate memmaps.
    """

    def setup_memory(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.accumulator = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=channels,
            filepath=accumulator_filepath,
            dtype=dtype,
        )
        if overlap_counter_filepath is not None:
            counter_filepath = overlap_counter_filepath
        elif accumulator_filepath is not None:
            suffix = accumulator_filepath.suffix
            counter_filepath = accumulator_filepath.with_suffix(f".overlaps{suffix}")
        else:
            counter_filepath = None
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=1,
            filepath=counter_filepath,
            dtype=dtype,
        )


class MaskBuilderFactory:
    """Factory that composes mask builders from capability flags.

    Simple and direct: specify what capabilities you need, and the factory
    builds the correct mixin hierarchy automatically.
    """

    _AGGREGATION_OPTIONS: dict[str, type[MaskBuilderABC]] = {
        "average": AveragingMaskBuilderMixin,
        "max": MaxMaskBuilderMixin,
    }

    @staticmethod
    def create(
        *,
        # Storage
        use_memmap: bool = False,
        # Aggregation
        aggregation: str = "average",
        # Coordinate scaling
        auto_scale: bool = False,
        # Tiling
        expand_scalars: bool = False,
        # Edge clipping
        px_to_clip: tuple[int, ...] | None = None,
        # Extra customization
        extra_mixins: Sequence[type[MaskBuilderABC]] | None = None,
    ) -> type[MaskBuilderABC]:
        """Create a mask builder class with specified capabilities.

        Args:
            use_memmap: Use memory-mapped storage (disk) instead of RAM arrays
            aggregation: Aggregation strategy ("average" or "max")
            auto_scale: Enable automatic coordinate scaling for resolution changes
            expand_scalars: Expand scalar predictions to uniform tile regions
            px_to_clip: Edge clipping specification as tuple of clip amounts per dimension
            extra_mixins: Additional custom mixins to include

        Returns:
            Composed mask builder class (not instantiated)
        """
        if aggregation not in MaskBuilderFactory._AGGREGATION_OPTIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. Supported: {tuple(MaskBuilderFactory._AGGREGATION_OPTIONS.keys())}."
            )

        # Build mixin list in correct MRO order
        mixins: list[type[MaskBuilderABC]] = []

        # 1. Storage (provides allocate_accumulator)
        if use_memmap:
            mixins.append(NumpyMemMapMaskBuilderAllocatorMixin)
        else:
            mixins.append(NumpyArrayMaskBuilderAllocatorMixin)

        # 2. Scaling (adjusts coordinates)
        if auto_scale:
            mixins.append(AutoScalingConstantStrideMixin)

        # 3. Clipping (modifies tiles and coordinates before aggregation)
        if px_to_clip is not None:
            mixins.append(EdgeClippingMaskBuilderMixin)

        # 4. Tiling (transforms input before aggregation)
        if expand_scalars:
            mixins.append(ScalarUniformTiledMaskBuilder)

        # 5. Aggregation (provides update_batch_impl and finalize)
        if aggregation == "average" and use_memmap:
            # Special memmap averaging handles overlap counter allocation
            aggregation_mixin = MemMapAveragingMaskBuilderMixin
        else:
            aggregation_mixin = MaskBuilderFactory._AGGREGATION_OPTIONS[aggregation]
        mixins.append(aggregation_mixin)

        # 6. Extra custom mixins
        if extra_mixins:
            mixins.extend(extra_mixins)

        # Compose class
        return type("MaskBuilder", tuple(mixins), {})
