from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from ratiopath.masks.mask_builders.aggregation import (
    Aggregator,
    MeanAggregator,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import pyvips


type EdgeClipping = int | tuple[int, ...] | tuple[tuple[int, int], ...]


@lru_cache(maxsize=32)
def _prepare_clipping(
    num_spatial_dims: int,
    tile_spatial_shape: tuple[int, ...],
    edge_clipping: EdgeClipping,
) -> tuple[tuple[slice, ...], NDArray[np.int64]]:
    """Calculate full slice objects and coordinate shifts for edge clipping."""
    if isinstance(edge_clipping, int):
        clip_start = np.full(num_spatial_dims, edge_clipping, dtype=np.int64)
        clip_end = np.full(num_spatial_dims, edge_clipping, dtype=np.int64)
    elif (
        isinstance(edge_clipping, tuple)
        and len(edge_clipping) == num_spatial_dims
        and all(
            isinstance(clip, tuple)
            and len(clip) == 2
            and all(isinstance(value, int) for value in clip)
            for clip in edge_clipping
        )
    ):
        clip_pairs = np.asarray(edge_clipping, dtype=np.int64)
        clip_start = clip_pairs[:, 0]
        clip_end = clip_pairs[:, 1]
    elif isinstance(edge_clipping, tuple) and len(edge_clipping) == num_spatial_dims:
        clip_start = np.asarray(edge_clipping, dtype=np.int64)
        clip_end = np.asarray(edge_clipping, dtype=np.int64)
    else:
        raise ValueError(
            "edge_clipping must be an int, "
            f"a tuple of {num_spatial_dims} ints, "
            f"or a tuple of {num_spatial_dims} (start, end) pairs."
        )

    slices = (
        slice(None),
        slice(None),
        *(
            slice(start, extent - end)
            for start, end, extent in zip(
                clip_start, clip_end, tile_spatial_shape, strict=True
            )
        ),
    )
    return slices, clip_start


class MaskBuilder[DType: np.generic, AggregatorR]:
    """Builder for assembling large masks from tiled data with automatic scaling and clipping.

    This class coordinates:
    - Coordinate scaling: Maps source WSI coordinates to mask resolution.
    - Edge clipping: Removes artifacts from model output tiles.
    - Pluggable storage: Allocates memory (RAM vs disk-backed).
    - Pluggable aggregation: Merges overlapping tiles (e.g., mean, max).
    """

    storage: NDArray[DType]
    aggregator: Aggregator[DType, AggregatorR]

    def __init__(
        self,
        source_extents: tuple[int, ...],
        source_tile_extent: int | tuple[int, ...],
        output_tile_extent: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        n_channels: int = 1,
        storage: Literal["inmemory", "memmap"]
        | Callable[..., NDArray[DType]] = "inmemory",
        aggregation: type[Aggregator[DType, AggregatorR]] = MeanAggregator,  # type: ignore[assignment]
        dtype: type[DType] = np.float32,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> None:
        """Initialize the mask builder.

        Args:
            source_extents: Spatial dimensions (H, W, ...) of the source WSI.
            source_tile_extent: Spatial dimensions of the model input tiles.
            output_tile_extent: Spatial dimensions of the model output tiles.
            stride: Stride between tiles in source resolution.
            n_channels: Number of channels in the output mask.
            storage: Strategy for allocating memory ("inmemory", "memmap", or a class).
            aggregation: Strategy for combining tiles ("mean", "max", or a class).
            dtype: Data type for the accumulator.
            kwargs: Extra arguments passed to storage and aggregation initialization.
        """
        # Normalize spatial dimensions to arrays
        self.source_extents = np.asarray(source_extents, dtype=np.int64)
        self.source_tile_extent = np.broadcast_to(
            source_tile_extent, len(source_extents)
        )
        self.output_tile_extent = np.broadcast_to(
            output_tile_extent, len(source_extents)
        )
        self.stride = np.broadcast_to(stride, len(source_extents))

        # Calculate how many tiles are required to fully cover the WSI (implicitly padding the edges)
        # Using np.ceil to ensure the right/bottom edges are fully covered.
        num_tiles = (
            np.ceil(
                np.maximum(0, self.source_extents - self.source_tile_extent)
                / self.stride
            ).astype(np.int64)
            + 1
        )

        # Calculate the actual spatial span these tiles cover in the original WSI space
        self.span = (num_tiles - 1) * self.stride + self.source_tile_extent

        # Find the Greatest Common Divisor for the alignment math
        gcd = np.gcd(self.source_tile_extent, self.stride * self.output_tile_extent)

        # Calculate the required MaskBuilder properties
        self.mask_extents = (self.span * self.output_tile_extent) // gcd
        self.upscale_factor = self.source_tile_extent // gcd
        self.mask_stride = (self.stride * self.output_tile_extent) // gcd

        # Initialize Storage
        if isinstance(storage, str):
            if storage == "inmemory":
                from ratiopath.masks.mask_builders.storage import InMemory

                storage = InMemory
            elif storage == "memmap":
                from ratiopath.masks.mask_builders.storage import MemMap

                storage = MemMap
            else:
                raise ValueError(f"Unknown storage type: {storage}")

        self.storage = storage(
            shape=(n_channels, *self.mask_extents), dtype=dtype, **kwargs
        )

        # Initialize Aggregator
        self.aggregator = aggregation(storage=self.storage, **kwargs)

    def update_batch(
        self,
        batch: NDArray[DType],
        coords: NDArray[np.int64],
        edge_clipping: EdgeClipping = 0,
    ) -> None:
        """Update the accumulator with a batch of tiles.

        Args:
            batch: Array of shape (B, C, *SpatialDims) or (B, C) containing B tiles.
            coords: Array of shape (B, N) containing top-left coordinates in source resolution.
            edge_clipping: Pixels to clip from tile edges.
                Supports an int (symmetric for all dims), a tuple of N ints
                (symmetric per dim), or a tuple of N (start, end) pairs.
        """
        # Scale coordinates from source to mask resolution
        # Find which tile index this corresponds to and multiply by the mask stride
        # to get the starting pixel in the mask array
        coords = (coords // self.stride) * self.mask_stride

        # Handle scalar data expansion
        # Broadcast scalar (B, C) to (B, C, *output_tile_extent)
        # If already dense, this is a fast view operation
        num_missing_dims = len(self.mask_extents) + 2 - batch.ndim
        batch = np.broadcast_to(
            batch[..., *[np.newaxis] * num_missing_dims],
            (*batch.shape[:2], *self.output_tile_extent),
        )

        # Apply Edge Clipping
        slices, shift = _prepare_clipping(
            len(self.mask_extents), tuple(self.output_tile_extent), edge_clipping
        )
        batch = batch[slices]
        coords += shift * self.upscale_factor

        # Upscale the batch to match the mask resolution if needed
        for axis_idx, factor in enumerate(self.upscale_factor, start=2):
            if factor > 1:
                batch = np.repeat(batch, factor, axis=axis_idx)

        for sample, coord in zip(batch, coords, strict=True):
            self.aggregator.update(self.storage, sample, coord)

    def finalize(self) -> AggregatorR:
        return self.aggregator.finalize(self.storage)

    def resize_to_source(self, image: NDArray[DType]) -> pyvips.Image:
        """Resize a mask array to the original source resolution using pyvips.

        Args:
            image: Array of shape (C, H_mask, W_mask) to be resized to (H_source, W_source, C).
        """
        import pyvips

        vips_image = pyvips.Image.new_from_array(image.transpose(1, 2, 0))

        scale_factors = self.source_extents / self.mask_extents
        vips_image = vips_image.resize(scale_factors[1], vscale=scale_factors[0])
        vips_image = vips_image.crop(
            0, 0, self.source_extents[1], self.source_extents[0]
        )

        return vips_image

    def cleanup(self) -> None:
        if hasattr(self, "storage"):
            if hasattr(self.storage, "close"):
                self.storage.close()
            del self.storage

        self.aggregator.cleanup()
