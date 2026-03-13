from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from ratiopath.masks.mask_builders.aggregation import Aggregator
from ratiopath.masks.mask_builders.receptive_field_manipulation import Preprocessor
from ratiopath.masks.mask_builders.storage import StorageAllocator


class MaskBuilder[DType: np.generic]:
    """Builder for assembling large masks from tiled data using a composed strategy pattern.

    Instead of relying on mixins, this class takes pluggable components:
    - storage: Determines how the accumulator memory is allocated (e.g. RAM vs disk).
    - aggregator: Determines how overlapping tiles are merged (e.g. max, mean).
    - preprocessors: A list of transformations applied to tiles/coords before accumulation.
    """

    @overload
    def __init__(
        self,
        shape: tuple[int, ...],
        storage: Literal["numpy", "memmap"] | type[StorageAllocator[DType]],
        aggregation: Literal["mean", "max"]
        | type[Aggregator[DType]]
        | Aggregator[DType] = "mean",
        preprocessors: Sequence[Preprocessor] = (),
        *,
        dtype: type[DType] = np.float32,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        shape: tuple[int, ...],
        storage: StorageAllocator[DType],
        aggregation: Literal["mean", "max"]
        | type[Aggregator[DType]]
        | Aggregator[DType] = "mean",
        preprocessors: Sequence[Preprocessor] = (),
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        shape: tuple[int, ...],
        storage: Literal["numpy", "memmap"]
        | type[StorageAllocator[DType]]
        | StorageAllocator[DType] = "numpy",
        aggregation: Literal["mean", "max"]
        | type[Aggregator[DType]]
        | Aggregator[DType] = "mean",
        preprocessors: Sequence[Preprocessor] = (),
        *,
        dtype: type[DType] = np.float32,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> None:
        """Initialize the mask builder.

        Args:
            shape: Spatial dimensions of the mask to build.
            storage: Strategy for allocating memory ("numpy", "memmap", a class, or an instance).
            aggregation: Strategy for combining tiles ("mean", "max", a class, or an instance).
            preprocessors: Optional sequence of preprocessing steps.
            dtype: Data type for the accumulator.
            **kwargs: Extra arguments passed to storage and aggregator initialization.
        """
        for preprocessor in preprocessors:
            shape = preprocessor.reshape_storage(shape)

        # Resolve Storage
        if isinstance(storage, str):
            if storage == "numpy":
                from ratiopath.masks.mask_builders.storage import NumpyStorage

                storage_cls: type[StorageAllocator[DType]] = NumpyStorage
            elif storage == "memmap":
                from ratiopath.masks.mask_builders.storage import MemMapStorage

                storage_cls = MemMapStorage
            else:
                raise ValueError(f"Unknown storage type: {storage}")

            self.storage = storage_cls(shape=shape, dtype=dtype, **kwargs)
        elif isinstance(storage, type) and issubclass(storage, StorageAllocator):
            self.storage = storage(shape=shape, dtype=dtype, **kwargs)
        else:
            self.storage = storage

        # Resolve Aggregator
        if isinstance(aggregation, str):
            if aggregation == "mean":
                from ratiopath.masks.mask_builders.aggregation import MeanAggregator

                aggregator_cls: type[Aggregator[DType]] = MeanAggregator
            elif aggregation == "max":
                from ratiopath.masks.mask_builders.aggregation import MaxAggregator

                aggregator_cls = MaxAggregator
            else:
                raise ValueError(f"Unknown aggregation type: {aggregation}")

            self.aggregator = aggregator_cls(
                storage=self.storage, shape=shape, dtype=dtype, **kwargs
            )
        elif isinstance(aggregation, type) and issubclass(aggregation, Aggregator):
            self.aggregator = aggregation(
                storage=self.storage,
                shape=shape,
                dtype=dtype,
                **kwargs,
            )
        else:
            self.aggregator = aggregation

        self.preprocessors = preprocessors

    def update_batch(self, data_batch: np.ndarray, coords_batch: np.ndarray) -> None:
        """Update the accumulator with a batch of tiles.

        Args:
            data_batch: Array of shape (B, C, *SpatialDims) or (B, C) containing B tiles.
            coords_batch: Array of shape (N, B) containing top-left coordinates.
        """
        for preprocessor in self.preprocessors:
            data_batch, coords_batch = preprocessor.process(data_batch, coords_batch)

        self.aggregator.update(self.storage.accumulator, data_batch, coords_batch)

    def finalize(self) -> dict[str, NDArray[DType]] | NDArray[DType]:
        """Finalize the mask and perform any necessary final computations (like averaging)."""
        return self.aggregator.finalize(self.storage.accumulator)
