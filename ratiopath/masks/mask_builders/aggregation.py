from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64, Shaped

from ratiopath.masks.mask_builders.mask_builder import (
    AccumulatorType,
    MaskBuilder,
    compute_acc_slices,
)


class AveragingMaskBuilderMixin(MaskBuilder):
    """Mixin that implements averaging aggregation for overlapping tiles.

    This mixin accumulates tiles by addition and tracks the overlap count at each pixel.
    During finalization, the accumulated values are divided by the overlap count to compute
    the average value at each position. This is useful for:
    - Smoothly blending overlapping tile predictions
    - Reducing edge artifacts in sliding window processing
    - Computing ensemble averages from multiple passes

    The mixin allocates an additional `overlap_counter` accumulator with shape (1, *SpatialDims)
    to track how many tiles contributed to each pixel position.
    """

    overlap_counter: AccumulatorType

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike,
        **kwargs: Any,
    ) -> None:
        super().__init__(mask_extents, channels, dtype=dtype, **kwargs)

    def setup_memory(
        self, mask_extents, channels, dtype: npt.DTypeLike, **kwargs
    ) -> None:
        """Set up memory for both the main accumulator and the overlap counter.

        Args:
            mask_extents: Array of shape (N,) specifying the spatial dimensions of the mask to build.
            channels: Number of channels in the mask (e.g., 1 for grayscale, 3 for RGB).
            dtype: Data type for the accumulators (e.g., np.float32).
            **kwargs: Additional keyword arguments passed to `allocate_accumulator()`.
        """
        # Perform base allocation then allocate the overlap counter.
        super().setup_memory(mask_extents, channels, dtype=dtype, **kwargs)
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents, channels=1, dtype=dtype, **kwargs
        )

    def update_batch_impl(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        mask_tile_extents = np.asarray(data_batch.shape[2:])  # H, W, ...
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            self.accumulator[
                :,
                *acc_slices,
            ] += data
            self.overlap_counter[
                :,
                *acc_slices,
            ] += 1

    def finalize(self) -> tuple[AccumulatorType, ...]:
        # Average the accumulated mask by the overlap counts
        self.accumulator /= self.overlap_counter.clip(min=1)
        return self.accumulator, self.overlap_counter


class MaxMaskBuilderMixin(MaskBuilder):
    """Mixin that implements maximum aggregation for overlapping tiles.

    This mixin keeps only the maximum value at each pixel position when tiles overlap.
    No additional storage is required, and finalization is a no-op since the accumulator
    already contains the final max values. This is useful for:
    - Maximum intensity projection
    - Keeping the highest confidence prediction across overlapping tiles
    - Peak detection across multiple scales
    """

    def update_batch_impl(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        mask_tile_extents = np.asarray(
            data_batch.shape[2:], dtype=np.int64
        )  # H, W, ...
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,  # H, W, ...
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            self.accumulator[
                :,
                *acc_slices,
            ] = np.maximum(
                self.accumulator[
                    :,
                    *acc_slices,
                ],
                data,
            )

    def finalize(self) -> tuple[AccumulatorType]:
        return (self.accumulator,)
