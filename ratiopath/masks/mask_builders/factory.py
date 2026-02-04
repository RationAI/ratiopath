from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.aggregation import (
    AveragingMaskBuilderMixin,
    MaxMaskBuilderMixin,
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


if TYPE_CHECKING:
    from pathlib import Path

    from ratiopath.masks.mask_builders.mask_builder import (
        AccumulatorType,
        MaskBuilderABC,
    )


# ============================================================================
# COMBINED MIXINS
# ============================================================================


class NumpyMemMapForMeanMaskBuilderMixin(
    NumpyMemMapMaskBuilderAllocatorMixin,
):
    """Mixin for memory-mapped averaging mask builders with dual memmap allocation.

    This mixin provides specialized setup_memory() logic for the case where both
    the main accumulator and the overlap counter need to be allocated as memory-mapped
    arrays.

    The overlap counter is automatically allocated with uint16 dtype and derives
    its filepath from the accumulator filepath with a .overlaps suffix when not
    explicitly provided.

    Aggregation logic (update_batch_impl, finalize) must be provided by including
    AveragingMaskBuilderMixin in the final class definition after this mixin.
    """

    overlap_counter: AccumulatorType

    def setup_memory(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up memory for both the main accumulator and the overlap counter.

        This method does not call super().setup_memory(), instead directly allocating
        both arrays using allocate_accumulator(). This is necessary because:
        1. Both arrays need memmap allocation (from NumpyMemMapMaskBuilderAllocatorMixin)
        2. The overlap counter needs special filepath logic (automatic .overlaps suffix)
        3. Calling super() would complicate the filepath passing through the MRO chain

        Args:
            mask_extents: Array of shape (N,) specifying the spatial dimensions of the mask to build.
            channels: Number of channels in the mask (e.g., 1 for grayscale, 3 for RGB).
            dtype: Data type for the accumulator (e.g., np.float32).
            accumulator_filepath: Optional Path to back the memmap file for the accumulator.
            overlap_counter_filepath: Optional Path to back the memmap file for the overlap counter.
            **kwargs: Additional keyword arguments for allocation.
        """
        # Allocate main accumulator as memmap
        self.accumulator = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=channels,
            filepath=accumulator_filepath,
            dtype=dtype,
        )

        # Generate overlap counter filepath if not provided
        if overlap_counter_filepath is not None:
            counter_filepath = overlap_counter_filepath
        elif accumulator_filepath is not None:
            suffix = accumulator_filepath.suffix
            counter_filepath = accumulator_filepath.with_suffix(f".overlaps{suffix}")
        else:
            counter_filepath = None

        # Allocate overlap counter as memmap with uint16 dtype
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=1,
            filepath=counter_filepath,
            dtype=np.uint16,
        )


class ScalarMeanMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    ScalarUniformTiledMaskBuilder,
    AveragingMaskBuilderMixin,
):
    """Scalar uniform tiled mask builder with auto-scaling and averaging aggregation.

    This builder combines:
    - **Scalar uniform tiling**: Expands scalar/vector values into uniform tiles
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging
    - **Numpy array storage**: Uses RAM-based accumulation

    Note: EdgeClipping is not available for scalar-based builders due to architectural constraints
    with GCD compression. Scalar data lacks spatial dimensions, preventing edge clipping.

    Args:
        source_extents: Spatial dimensions of the source image.
        source_tile_extents: Size of input tiles extracted from source.
        source_tile_strides: Stride between input tiles in source space.
        mask_tile_extents: Size you want each scalar to represent in output mask.
        channels: Number of channels in scalar predictions.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            dtype=dtype,
            **kwargs,
        )


class ScalarMaxMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    ScalarUniformTiledMaskBuilder,
    MaxMaskBuilderMixin,
):
    """Scalar uniform tiled mask builder with auto-scaling and max aggregation.

    This builder combines:
    - **Scalar uniform tiling**: Expands scalar/vector values into uniform tiles
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)
    - **Numpy array storage**: Uses RAM-based accumulation

    Note: EdgeClipping is not available for scalar-based builders due to architectural constraints
    with GCD compression. Scalar data lacks spatial dimensions, preventing edge clipping.

    Args:
        source_extents: Spatial dimensions of the source image.
        source_tile_extents: Size of input tiles extracted from source.
        source_tile_strides: Stride between input tiles in source space.
        mask_tile_extents: Size you want each scalar to represent in output mask.
        channels: Number of channels in scalar predictions.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            dtype=dtype,
            **kwargs,
        )


class MemMapMeanMB(
    NumpyMemMapForMeanMaskBuilderMixin,
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    AveragingMaskBuilderMixin,
):
    """Memory-mapped averaging mask builder with auto-scaling and edge clipping support.

    This builder combines:
    - **Memory-mapped storage**: Handles large masks that exceed RAM capacity using disk-backed arrays
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging

    Allocates two memmaps:
    - Main accumulator for tile data
    - Overlap counter (auto-derived with `.overlaps` suffix if accumulator_filepath is provided)

    Args:
        source_extents: Spatial dimensions of the source space.
        source_tile_extents: Extents of tiles in the source space.
        source_tile_strides: Strides between tiles in the source space.
        mask_tile_extents: Extents of tiles in the mask space.
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        accumulator_filepath: Optional path for the main accumulator memmap.
        overlap_counter_filepath: Optional path for the overlap counter memmap.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            accumulator_filepath=accumulator_filepath,
            overlap_counter_filepath=overlap_counter_filepath,
            dtype=dtype,
            **kwargs,
        )


class MemMapMaxMB(
    NumpyMemMapMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    MaxMaskBuilderMixin,
):
    """Memory-mapped max mask builder with auto-scaling and edge clipping support.

    This builder combines:
    - **Memory-mapped storage**: Handles large masks that exceed RAM capacity using disk-backed arrays
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)

    Args:
        source_extents: Spatial dimensions of the source space.
        source_tile_extents: Extents of tiles in the source space.
        source_tile_strides: Strides between tiles in the source space.
        mask_tile_extents: Extents of tiles in the mask space.
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        accumulator_filepath: Optional path for the main accumulator memmap.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        accumulator_filepath: Path | None = None,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            accumulator_filepath=accumulator_filepath,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordMemMapMeanMB(
    NumpyMemMapForMeanMaskBuilderMixin,
    EdgeClippingMaskBuilderMixin,
    AveragingMaskBuilderMixin,
):
    """Memory-mapped averaging mask builder with explicit coordinates and edge clipping support.

    This builder combines:
    - **Memory-mapped storage**: Handles large masks that exceed RAM capacity using disk-backed arrays
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging

    Allocates two memmaps:
    - Main accumulator for tile data
    - Overlap counter (auto-derived with `.overlaps` suffix if accumulator_filepath is provided)

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        accumulator_filepath: Optional path for the main accumulator memmap.
        overlap_counter_filepath: Optional path for the overlap counter memmap.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            accumulator_filepath=accumulator_filepath,
            overlap_counter_filepath=overlap_counter_filepath,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordMemMapMaxMB(
    NumpyMemMapMaskBuilderAllocatorMixin,
    EdgeClippingMaskBuilderMixin,
    MaxMaskBuilderMixin,
):
    """Memory-mapped max mask builder with explicit coordinates and edge clipping support.

    This builder combines:
    - **Memory-mapped storage**: Handles large masks that exceed RAM capacity using disk-backed arrays
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        accumulator_filepath: Optional path for the main accumulator memmap.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        accumulator_filepath: Path | None = None,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            accumulator_filepath=accumulator_filepath,
            dtype=dtype,
            **kwargs,
        )


class MeanMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    AveragingMaskBuilderMixin,
):
    """Tile-based averaging mask builder with auto-scaling and edge clipping support.

    This builder combines:
    - **Numpy array storage**: Uses RAM-based accumulation
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging

    Allocates two numpy arrays:
    - Main accumulator for tile data
    - Overlap counter for tracking tile overlaps

    Args:
        source_extents: Spatial dimensions of the source space.
        source_tile_extents: Extents of tiles in the source space.
        source_tile_strides: Strides between tiles in the source space.
        mask_tile_extents: Extents of tiles in the mask space.
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            dtype=dtype,
            **kwargs,
        )


class MaxMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilderMixin,
    MaxMaskBuilderMixin,
):
    """Tile-based max mask builder with auto-scaling and edge clipping support.

    This builder combines:
    - **Numpy array storage**: Uses RAM-based accumulation
    - **Auto-scaling**: Automatically handles coordinate transformation for resolution changes
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)

    Args:
        source_extents: Spatial dimensions of the source space.
        source_tile_extents: Extents of tiles in the source space.
        source_tile_strides: Strides between tiles in the source space.
        mask_tile_extents: Extents of tiles in the mask space.
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordMeanMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    EdgeClippingMaskBuilderMixin,
    AveragingMaskBuilderMixin,
):
    """Tile-based averaging mask builder with explicit coordinates and edge clipping support.

    This builder combines:
    - **Numpy array storage**: Uses RAM-based accumulation
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging

    Allocates two numpy arrays:
    - Main accumulator for tile data
    - Overlap counter for tracking tile overlaps

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordMaxMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    EdgeClippingMaskBuilderMixin,
    MaxMaskBuilderMixin,
):
    """Tile-based max mask builder with explicit coordinates and edge clipping support.

    This builder combines:
    - **Numpy array storage**: Uses RAM-based accumulation
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Edge clipping**: Optional removal of boundary artifacts (default: no clipping)
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the tiles/mask.
        px_to_clip: Edge clipping specification (default: 0, no clipping).
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            px_to_clip=px_to_clip,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordScalarMeanMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    ScalarUniformTiledMaskBuilder,
    AveragingMaskBuilderMixin,
):
    """Scalar uniform tiled mask builder with explicit coordinates.

    This builder combines:
    - **Scalar uniform tiling**: Expands scalar/vector values into uniform tiles
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Averaging aggregation**: Smoothly blends overlapping tiles via pixel-wise averaging
    - **Numpy array storage**: Uses RAM-based accumulation

    Note: EdgeClipping is not available for scalar-based builders due to architectural constraints
    with GCD compression. Scalar data lacks spatial dimensions, preventing edge clipping.

    Use this builder when:
    - You know the exact output mask dimensions upfront
    - Your tiles have constant extents and strides (uniform tiling)
    - You want to average overlapping regions

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the scalar values.
        mask_tile_extents: Extents of the tiles in mask space.
        mask_tile_strides: Strides of the tiles in mask space.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            dtype=dtype,
            **kwargs,
        )


class ExplicitCoordScalarMaxMB(
    NumpyArrayMaskBuilderAllocatorMixin,
    ScalarUniformTiledMaskBuilder,
    MaxMaskBuilderMixin,
):
    """Scalar uniform tiled mask builder with explicit coordinates and max aggregation.

    This builder combines:
    - **Scalar uniform tiling**: Expands scalar/vector values into uniform tiles
    - **Explicit coordinates**: User must specify mask_extents directly (no auto-scaling)
    - **Max aggregation**: Takes maximum value at each pixel position (no finalization needed)
    - **Numpy array storage**: Uses RAM-based accumulation

    Note: EdgeClipping is not available for scalar-based builders due to architectural constraints
    with GCD compression. Scalar data lacks spatial dimensions, preventing edge clipping.

    Use this builder when:
    - You know the exact output mask dimensions upfront
    - Your tiles have constant extents and strides (uniform tiling)
    - You want to take the maximum value at overlapping regions

    Args:
        mask_extents: Spatial dimensions of the mask to build (must be provided by user).
        channels: Number of channels in the scalar values.
        mask_tile_extents: Extents of the tiles in mask space.
        mask_tile_strides: Strides of the tiles in mask space.
        dtype: Data type for the accumulator.
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            dtype=dtype,
            **kwargs,
        )


class MaskBuilderFactory:
    """Factory that composes mask builders from capability flags.

    Simple and direct: specify what capabilities you need, and the factory
    builds the correct mixin hierarchy automatically.
    """

    _AGGREGATION_OPTIONS: ClassVar[set[str]] = {
        "mean",
        "max",
    }

    @staticmethod
    def create(
        *,
        # Storage
        use_memmap: bool = False,
        # Aggregation
        aggregation: str = "mean",
        # Coordinate scaling
        auto_scale: bool = False,
        # Tiling
        expand_scalars: bool = False,
    ) -> type[MaskBuilderABC]:
        """Create a mask builder class with specified capabilities.

        EdgeClipping is only available for tile-based builders (expand_scalars=False), not scalar-based
        builders (expand_scalars=True), due to architectural constraints with GCD compression.

        Supported combinations:
        Scalar-based (expand_scalars=True):
          1. auto_scale=True, aggregation="mean" -> ScalarMeanMB
          2. auto_scale=True, aggregation="max" -> ScalarMaxMB
          3. auto_scale=False, aggregation="mean" -> ExplicitCoordScalarMeanMB
          4. auto_scale=False, aggregation="max" -> ExplicitCoordScalarMaxMB

        Tile-based with memmap (use_memmap=True, expand_scalars=False):
          5. auto_scale=True, aggregation="mean" -> MemMapMeanMB
          6. auto_scale=True, aggregation="max" -> MemMapMaxMB
          7. auto_scale=False, aggregation="mean" -> ExplicitCoordMemMapMeanMB
          8. auto_scale=False, aggregation="max" -> ExplicitCoordMemMapMaxMB

        Tile-based with numpy (use_memmap=False, expand_scalars=False):
          9. auto_scale=True, aggregation="mean" -> MeanMB
          10. auto_scale=True, aggregation="max" -> MaxMB
          11. auto_scale=False, aggregation="mean" -> ExplicitCoordMeanMB
          12. auto_scale=False, aggregation="max" -> ExplicitCoordMaxMB

        Args:
            use_memmap: Use memory-mapped storage (disk) instead of RAM arrays
            aggregation: Aggregation strategy ("mean" or "max")
            auto_scale: Enable automatic coordinate scaling for resolution changes. If True, mask extents are computed automatically
                based on source extents, source tile extents, and source tile strides, and coordinates are automatically scaled from source space to mask space,
                without user intervention. For more details, see `AutoScalingConstantStrideMixin`.
                If False, user must provide explicit mask extents and coordinates are used as-is.
            expand_scalars: Expand scalar predictions to uniform tile regions. If True,
                the builder expects scalar/vector values and expands them into uniform tiles over the full extent of the tile,
                extents of which are specified as a parameter.

        Returns:
            One of the predefined mask builder classes (not instantiated)

        Raises:
            ValueError: If the parameter combination does not match a predefined configuration
        """
        if aggregation not in MaskBuilderFactory._AGGREGATION_OPTIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. Supported: {tuple(MaskBuilderFactory._AGGREGATION_OPTIONS)}."
            )

        # Case 1: auto_scale + expand_scalars + mean (numpy)
        if not use_memmap and aggregation == "mean" and auto_scale and expand_scalars:
            return ScalarMeanMB

        # Case 2: auto_scale + expand_scalars + max (numpy)
        if not use_memmap and aggregation == "max" and auto_scale and expand_scalars:
            return ScalarMaxMB

        # Case 3: use_memmap + auto_scale + mean
        if use_memmap and aggregation == "mean" and auto_scale and not expand_scalars:
            return MemMapMeanMB

        # Case 4: explicit coordinates + expand_scalars + mean (numpy)
        if (
            not use_memmap
            and aggregation == "mean"
            and not auto_scale
            and expand_scalars
        ):
            return ExplicitCoordScalarMeanMB

        # Case 5: explicit coordinates + expand_scalars + max (numpy)
        if (
            not use_memmap
            and aggregation == "max"
            and not auto_scale
            and expand_scalars
        ):
            return ExplicitCoordScalarMaxMB

        # Case 6: use_memmap + auto_scale + max
        if use_memmap and aggregation == "max" and auto_scale and not expand_scalars:
            return MemMapMaxMB

        # Case 7: use_memmap + explicit coordinates + mean
        if (
            use_memmap
            and aggregation == "mean"
            and not auto_scale
            and not expand_scalars
        ):
            return ExplicitCoordMemMapMeanMB

        # Case 8: use_memmap + explicit coordinates + max
        if (
            use_memmap
            and aggregation == "max"
            and not auto_scale
            and not expand_scalars
        ):
            return ExplicitCoordMemMapMaxMB

        # Case 9: numpy + auto_scale + mean (tile-based)
        if (
            not use_memmap
            and aggregation == "mean"
            and auto_scale
            and not expand_scalars
        ):
            return MeanMB

        # Case 10: numpy + auto_scale + max (tile-based)
        if (
            not use_memmap
            and aggregation == "max"
            and auto_scale
            and not expand_scalars
        ):
            return MaxMB

        # Case 11: numpy + explicit coordinates + mean (tile-based)
        if (
            not use_memmap
            and aggregation == "mean"
            and not auto_scale
            and not expand_scalars
        ):
            return ExplicitCoordMeanMB

        # Case 12: numpy + explicit coordinates + max (tile-based)
        if (
            not use_memmap
            and aggregation == "max"
            and not auto_scale
            and not expand_scalars
        ):
            return ExplicitCoordMaxMB

        # If no predefined combination matches, raise an error
        raise ValueError(
            f"Unsupported combination of parameters: "
            f"use_memmap={use_memmap}, aggregation={aggregation}, "
            f"auto_scale={auto_scale}, expand_scalars={expand_scalars}. "
            f"Supported combinations are:\n"
            f"Scalar-based (expand_scalars=True):\n"
            f"  1. auto_scale=True, aggregation='mean' -> ScalarMeanMB\n"
            f"  2. auto_scale=True, aggregation='max' -> ScalarMaxMB\n"
            f"  3. auto_scale=False, aggregation='mean' -> ExplicitCoordScalarMeanMB\n"
            f"  4. auto_scale=False, aggregation='max' -> ExplicitCoordScalarMaxMB\n"
            f"Tile-based with memmap (use_memmap=True, expand_scalars=False):\n"
            f"  5. auto_scale=True, aggregation='mean' -> MemMapMeanMB\n"
            f"  6. auto_scale=True, aggregation='max' -> MemMapMaxMB\n"
            f"  7. auto_scale=False, aggregation='mean' -> ExplicitCoordMemMapMeanMB\n"
            f"  8. auto_scale=False, aggregation='max' -> ExplicitCoordMemMapMaxMB\n"
            f"Tile-based with numpy (use_memmap=False, expand_scalars=False):\n"
            f"  9. auto_scale=True, aggregation='mean' -> MeanMB\n"
            f"  10. auto_scale=True, aggregation='max' -> MaxMB\n"
            f"  11. auto_scale=False, aggregation='mean' -> ExplicitCoordMeanMB\n"
            f"  12. auto_scale=False, aggregation='max' -> ExplicitCoordMaxMB"
        )
