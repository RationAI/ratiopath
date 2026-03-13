from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ratiopath.masks.mask_builders.storage import StorageAllocator


def compute_acc_slices(
    coords_batch: np.ndarray, mask_tile_extents: np.ndarray
) -> list[list[slice]]:
    """Compute slice objects for accumulator indexing."""
    tile_end_coords = coords_batch + mask_tile_extents[:, np.newaxis]  # shape (N, B)

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


class MeanAggregator[DType: np.generic](Aggregator[DType]):
    """Aggregator that implements averaging aggregation for overlapping tiles."""

    def __init__(
        self,
        storage: StorageAllocator[DType],
        shape: tuple[int, ...],
        filename: Path | str | None = None,
        overlap_counter_filename: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        overlap_filename = overlap_counter_filename
        if overlap_filename is None and filename is not None:
            path = Path(filename)
            overlap_filename = path.with_suffix(f".overlaps{path.suffix}")

        # Instantiate the storage class again for the overlap counter
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("dtype", None)

        storage_cls = cast("type[StorageAllocator[np.uint16]]", type(storage))
        self.overlap_storage = storage_cls(
            filename=overlap_filename,
            shape=(1, *shape[1:]),
            dtype=np.uint16,
            **kwargs_copy,
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
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            accumulator[:, *acc_slices] += data  # type: ignore[misc]
            self.overlap_storage.accumulator[:, *acc_slices] += 1

    def finalize(self, accumulator: NDArray[DType]) -> dict[str, NDArray[Any]]:
        accumulator /= self.overlap_storage.accumulator.clip(min=1)  # type: ignore[misc]
        return {
            "mask": accumulator,
            "overlap_counter": self.overlap_storage.accumulator,
        }


class MaxAggregator[DType: np.generic](Aggregator[DType]):
    """Aggregator that implements maximum aggregation for overlapping tiles."""

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
