from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ratiopath.misc import safely_instantiate


def compute_acc_slices(
    coords_batch: np.ndarray, mask_tile_extents: np.ndarray
) -> list[list[slice]]:
    """Compute slice objects for accumulator indexing.

    Args:
        coords_batch: Array of shape (N, B) with top-left coordinates for B tiles in N dimensions.
        mask_tile_extents: Array of shape (N,) with tile size in mask space for each dimension.

    Returns:
        List of N lists, each containing B slice objects for indexing into accumulator.
    """
    tile_end_coords = coords_batch + mask_tile_extents[:, np.newaxis]

    acc_slices_batch_per_dim = []
    for dimension in range(coords_batch.shape[0]):
        acc_slices_batch_per_dim.append(
            [
                slice(start, end)
                for start, end in zip(
                    coords_batch[dimension], tile_end_coords[dimension], strict=True
                )
            ]
        )
    return acc_slices_batch_per_dim


class Aggregator[DType: np.generic](ABC):
    """Abstract base class for aggregation strategies."""

    def __init__(self, storage: NDArray[DType], **kwargs: Any) -> None:
        return

    @abstractmethod
    def update(
        self,
        accumulator: NDArray[DType],
        data_batch: np.ndarray,
        coords_batch: np.ndarray,
    ) -> None:
        """Update the accumulator with a batch of tiles."""

    @abstractmethod
    def finalize(
        self, accumulator: NDArray[DType]
    ) -> dict[str, NDArray[Any]] | NDArray[DType]:
        """Finalize the mask assembly and return the result."""

    def cleanup(self) -> None:
        """Optional cleanup method to release resources if needed."""
        return


class MeanAggregator[DType: np.generic](Aggregator[DType]):
    """Aggregator that implements averaging aggregation for overlapping tiles.

    This aggregator accumulates tiles by addition and tracks the overlap count at each pixel.
    During finalization, the accumulated values are divided by the overlap count to compute
    the average value at each position. This is useful for:
    - Smoothly blending overlapping tile predictions
    - Reducing edge artifacts in sliding window processing
    - Computing ensemble averages from multiple passes

    The aggregator allocates an additional `overlap_counter` accumulator with shape (1, *SpatialDims)
    to track how many tiles contributed to each pixel position.
    """

    def __init__(
        self,
        storage: NDArray[DType],
        filename: Path | str | None = None,
        overlap_counter_filename: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        overlap_filename = overlap_counter_filename
        if overlap_filename is None and filename is not None:
            path = Path(filename)
            overlap_filename = path.with_suffix(f".overlaps{path.suffix}")

        storage_cls = cast("type[NDArray[np.uint16]]", type(storage))
        self.overlap_counter = safely_instantiate(
            storage_cls,
            filename=overlap_filename,
            shape=(1, *storage.shape[1:]),
            dtype=np.uint16,
            **kwargs,
        )

    def update(
        self,
        accumulator: NDArray[DType],
        data_batch: np.ndarray,
        coords_batch: np.ndarray,
    ) -> None:
        mask_tile_extents = np.asarray(data_batch.shape[2:], dtype=np.int64)
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True), data_batch, strict=True
        ):
            accumulator[:, *acc_slices] += data  # type: ignore[misc]
            self.overlap_counter[:, *acc_slices] += 1

    def finalize(self, accumulator: NDArray[DType]) -> dict[str, NDArray[Any]]:
        accumulator /= self.overlap_counter.clip(min=1)  # type: ignore[misc]
        return {
            "mask": accumulator,
            "overlap_counter": self.overlap_counter,
        }

    def cleanup(self) -> None:
        if hasattr(self, "overlap_counter"):
            if hasattr(self.overlap_counter, "close"):
                self.overlap_counter.close()
            del self.overlap_counter


class MaxAggregator[DType: np.generic](Aggregator[DType]):
    """Aggregator that implements maximum aggregation for overlapping tiles.

    This aggregator keeps only the maximum value at each pixel position when tiles overlap.
    No additional storage is required, and finalization is a no-op since the accumulator
    already contains the final max values. This is useful for:
    - Maximum intensity projection
    - Keeping the highest confidence prediction across overlapping tiles
    - Peak detection across multiple scales
    """

    def update(
        self,
        accumulator: NDArray[DType],
        data_batch: np.ndarray,
        coords_batch: np.ndarray,
    ) -> None:
        mask_tile_extents = np.asarray(data_batch.shape[2:], dtype=np.int64)
        acc_slices_all_dims = compute_acc_slices(coords_batch, mask_tile_extents)

        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True), data_batch, strict=True
        ):
            accumulator[:, *acc_slices] = np.maximum(accumulator[:, *acc_slices], data)

    def finalize(self, accumulator: NDArray[DType]) -> NDArray[DType]:
        return accumulator
