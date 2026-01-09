import logging
import math
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
from jaxtyping import Int64, Shaped


logger = logging.getLogger(__name__)


AccumulatorType = np.ndarray

SpatialDims = TypeVar("SpatialDims", bound=tuple[int, ...])


def compute_acc_slices(
    coords_batch: Int64[AccumulatorType, "N B"],
    mask_tile_extents: Int64[AccumulatorType, " N"],
) -> list[list[slice]]:
    """Compute slice objects for accumulator indexing.

    Args:
        coords_batch: Array of shape (N, B) with top-left coordinates for B tiles in N dimensions.
        mask_tile_extents: Array of shape (N,) with tile size in mask space for each dimension.

    Returns:
        List of N lists, each containing B slice objects for indexing into accumulator.
    """
    acc_ends = coords_batch + mask_tile_extents[:, np.newaxis]  # shape (N, B)
    
    acc_slices_batch_per_dim = []
    for dimension in range(coords_batch.shape[0]):
        acc_slices_batch_per_dim.append([
            slice(start, end) 
            for start, end 
            in zip(coords_batch[dimension], acc_ends[dimension], strict=True)
        ])
    return acc_slices_batch_per_dim


class MaskBuilder(ABC):
    """Abstract base class for building masks from tiled data.

    This base class establishes the interface for mask builders that assemble large masks
    from batches of tiles. It uses a cooperative multiple inheritance pattern where:
    - `update_batch()` is concrete and can be wrapped by mixins
    - `update_batch_impl()` is abstract and must be implemented by concrete builders
    - `allocate_accumulator()` is abstract and defines storage strategy (numpy array, memmap, etc.)
    - `finalize()` is abstract and defines how to produce the final mask

    Subclasses can be composed using mixins to add features like edge clipping,
    averaging, max aggregation, etc. Mixins should override `update_batch()` and call
    `super().update_batch()` to maintain the cooperative chain.
    """

    accumulator: AccumulatorType

    def __init__(
        self, mask_extents: Int64[AccumulatorType, " N"], channels: int, **kwargs: Any
    ) -> None:
        """Initialize the mask builder and allocate the accumulator.

        Args:
            mask_extents: Array of shape (N,) specifying the spatial dimensions of the mask to build.
            channels: Number of channels in the mask (e.g., 1 for grayscale, 3 for RGB).
            **kwargs: Additional keyword arguments passed to `allocate_accumulator()`.
        """
        super().__init__()
        self.setup_memory(mask_extents, channels, **kwargs)

    @abstractmethod
    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: np.dtype,
        **kwargs: Any,
    ) -> AccumulatorType:
        """Allocates the necessary accumulators for assembling the mask."""
        ...

    def setup_memory(self, mask_extents, channels, **kwargs) -> None:
        """This method sets up memory structures needed for mask building.
        
        This methods can be overridden by mixins or concrete builders to set up any necessary memory structures.
        
        Some builders may require additional accumulators or data structures beyond the main accumulator.
        Some mixins may require temporary files for memory-mapped storage.
        All such setup should be defined in an overridden version of this method, which will be called by the base constructor
        after the initialisation parameters are set by all classes/mixins in the MRO chain.
        """
        self.accumulator = self.allocate_accumulator(
            mask_extents, channels, **kwargs
        )

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, " N B"],
    ) -> None:
        """Update the accumulator with a batch of tiles.

        This concrete method provides a stable entry point for mixins to wrap and extend
        functionality. It delegates to `update_batch_impl()` which must be implemented by
        concrete builders. Mixins can override this method and call `super().update_batch(...)`
        to form a cooperative chain of processing steps (e.g., edge clipping, coordinate adjustment)
        before the tiles reach the final implementation.

        This design allows unlimited stacking of mixins while avoiding issues with abstract methods
        in the MRO chain.

        Args:
            data_batch: Array of shape (B, C, *SpatialDims) containing B tiles with C channels.
            coords_batch: Array of shape (N, B) containing the top-left corner coordinates
                for each of the B tiles in N spatial dimensions.
        """
        return self.update_batch_impl(data_batch=data_batch, coords_batch=coords_batch)

    @abstractmethod
    def update_batch_impl(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, " N B"],
    ) -> None:
        """Core implementation for updating the accumulator with a batch of tiles.

        Concrete builders must implement this method with the actual logic for accumulating
        tiles into the mask. Common strategies include:
        - Addition (for averaging later)
        - Maximum (for max pooling)
        - Other aggregation operations

        This method is called by `update_batch()` after any mixin preprocessing has occurred.

        Args:
            data_batch: Array of shape (B, C, *SpatialDims) containing B tiles with C channels.
            coords_batch: Array of shape (N, B) containing the top-left corner coordinates
                for each of the B tiles in N spatial dimensions.
        """
        ...

    @abstractmethod
    def finalize(self) -> tuple[AccumulatorType, ...] | AccumulatorType:
        """Finalize the mask assembly and return the result.

        This method performs any necessary post-processing on the accumulator(s) and returns
        the final mask. Common operations include:
        - Averaging by overlap counts (for averaging builders)
        - No-op (for max builders where the accumulator is already final)
        - Other normalization or scaling operations

        Returns:
            Tuple of arrays where the first element is always the finalized mask.
            Additional elements may include auxiliary data like overlap counters.
        """
        ...


