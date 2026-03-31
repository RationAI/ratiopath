from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from collections.abc import Callable


class Aggregator[DType: np.generic, R](ABC):
    """Abstract base class for aggregation strategies."""

    def __init__(self, storage: NDArray[DType], **kwargs: Any) -> None:
        return

    @abstractmethod
    def update(
        self, accumulator: NDArray[DType], sample: np.ndarray, coords: NDArray[np.int64]
    ) -> None:
        """Update the accumulator with a single tile sample."""

    @abstractmethod
    def finalize(self, accumulator: NDArray[DType]) -> R:
        """Finalize the mask assembly and return the result."""

    def cleanup(self) -> None:
        """Optional cleanup method to release resources if needed."""
        return

    def _get_acc_slices(
        self, coords: NDArray[np.int64], mask_tile_extents: NDArray[np.int64]
    ) -> tuple[slice, ...]:
        """Compute slice objects for accumulator indexing.

        Args:
            coords: Array of shape (N,) with top-left coordinates in N dimensions.
            mask_tile_extents: Array of shape (N,) with tile size in mask space for each dimension.

        Returns:
            Tuple containing N slice objects for indexing into the accumulator.
        """
        tile_end_coords = coords + mask_tile_extents
        return tuple(
            slice(int(start), int(end))
            for start, end in zip(coords, tile_end_coords, strict=True)
        )


class MeanAggregatorResults[Dtype: np.generic](TypedDict):
    mask: NDArray[Dtype]
    overlap_counter: NDArray[np.uint16]


class MeanAggregator[DType: np.generic](
    Aggregator[DType, MeanAggregatorResults[DType]]
):
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

        storage_cls = cast("Callable[..., NDArray[np.uint16]]", type(storage))
        self.overlap_counter = storage_cls(
            filename=overlap_filename,
            shape=(1, *storage.shape[1:]),
            dtype=np.uint16,
            **kwargs,
        )

    def update(
        self, accumulator: NDArray[DType], sample: np.ndarray, coords: NDArray[np.int64]
    ) -> None:
        mask_tile_extents = np.asarray(sample.shape[1:], dtype=np.int64)
        acc_slices = self._get_acc_slices(coords, mask_tile_extents)
        accumulator[:, *acc_slices] += sample  # type: ignore[misc]
        self.overlap_counter[:, *acc_slices] += 1

    def finalize(self, accumulator: NDArray[DType]) -> MeanAggregatorResults[DType]:
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


class MaxAggregator[DType: np.generic](Aggregator[DType, NDArray[DType]]):
    """Aggregator that implements maximum aggregation for overlapping tiles.

    This aggregator keeps only the maximum value at each pixel position when tiles overlap.
    No additional storage is required, and finalization is a no-op since the accumulator
    already contains the final max values. This is useful for:
    - Maximum intensity projection
    - Keeping the highest confidence prediction across overlapping tiles
    - Peak detection across multiple scales
    """

    def update(
        self, accumulator: NDArray[DType], sample: np.ndarray, coords: NDArray[np.int64]
    ) -> None:
        mask_tile_extents = np.asarray(sample.shape[1:], dtype=np.int64)
        acc_slices = self._get_acc_slices(coords, mask_tile_extents)
        accumulator[:, *acc_slices] = np.maximum(accumulator[:, *acc_slices], sample)

    def finalize(self, accumulator: NDArray[DType]) -> NDArray[DType]:
        return accumulator
