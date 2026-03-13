from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    """Abstract base class for data and coordinate preprocessing steps."""

    def reshape_storage(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get the adjusted shape for storage based on the preprocessor's transformations."""
        return shape

    @abstractmethod
    def process(
        self, data_batch: np.ndarray, coords_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a batch of data and coordinates before accumulation.

        Args:
            data_batch: Batch of tile data of shape (B, C, *SpatialDims) or (B, C).
            coords_batch: Batch of tile coordinates of shape (N, B).

        Returns:
            Tuple of (processed_data_batch, processed_coords_batch).
        """


class EdgeClippingPreprocessor(Preprocessor):
    """Preprocessor that clips edge pixels from tiles before accumulation."""

    def __init__(self, px_to_clip: int | tuple[int, ...], num_dims: int) -> None:
        """Initialize the edge clipping preprocessor."""
        if isinstance(px_to_clip, int):
            clip_start_indices = (px_to_clip,) * num_dims
            clip_end_indices = (px_to_clip,) * num_dims
        elif isinstance(px_to_clip, tuple) and len(px_to_clip) == num_dims:
            clip_start_indices = px_to_clip
            clip_end_indices = px_to_clip
        elif isinstance(px_to_clip, tuple) and len(px_to_clip) == 2 * num_dims:
            clip_start_indices = px_to_clip[::2]
            clip_end_indices = px_to_clip[1::2]
        else:
            raise ValueError(
                f"px_to_clip must be an int, a tuple of {num_dims} ints, or a tuple of {2 * num_dims} ints."
            )

        self.clip_start_indices = np.asarray(clip_start_indices, dtype=np.int64)
        self.clip_end_indices = np.asarray(clip_end_indices, dtype=np.int64)

    def process(
        self, data_batch: np.ndarray, coords_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        extents = np.asarray(data_batch.shape[2:], dtype=np.int64)  # H, W, ...
        slices = tuple(
            slice(start, end)
            for start, end in zip(
                self.clip_start_indices, extents - self.clip_end_indices, strict=True
            )
        )
        adjusted_coords_batch = coords_batch + self.clip_start_indices[:, np.newaxis]
        clipped_data = data_batch[..., *slices]  # type: ignore[index, arg-type]
        return clipped_data, adjusted_coords_batch


class AutoScalingPreprocessor(Preprocessor):
    """Preprocessor that scales coordinates to match input/output resolution differences."""

    def __init__(
        self,
        source_extents: np.ndarray,
        source_tile_extents: np.ndarray,
        source_tile_strides: np.ndarray,
        mask_tile_extents: np.ndarray,
    ) -> None:
        self.source_tile_extents = np.asarray(source_tile_extents, dtype=np.int64)
        self.mask_tile_extents = np.asarray(mask_tile_extents, dtype=np.int64)
        source_tile_strides = np.asarray(source_tile_strides, dtype=np.int64)

        multiplied_ = source_tile_strides * self.mask_tile_extents
        if not np.all(multiplied_ % self.source_tile_extents == 0):
            raise ValueError(
                f"source_tile_strides * mask_tile_extents must be divisible by source_tile_extents in all dimensions,"
                f" but {source_tile_strides}*{self.mask_tile_extents}={multiplied_}, which is not divisible by {self.source_tile_extents}."
            )
        adjusted_mask_tile_strides = multiplied_ // self.source_tile_extents

        total_strides = (
            source_extents - self.source_tile_extents
        ) / source_tile_strides
        total_strides = np.ceil(total_strides).astype(np.int64)

        self.overflow_buffered_source_extents = (
            total_strides * source_tile_strides
        ) + self.source_tile_extents

        self.mask_extents = (
            total_strides * adjusted_mask_tile_strides
        ) + self.mask_tile_extents

    def process(
        self, data_batch: np.ndarray, coords_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adjusted_coords_batch = (
            coords_batch * self.mask_tile_extents[:, np.newaxis]
        ) // self.source_tile_extents[:, np.newaxis]
        return data_batch, adjusted_coords_batch

    def get_vips_scale_factors(
        self, accumulator_shape: tuple[int, ...]
    ) -> tuple[float, float]:
        """Get the scaling factors to convert the built mask back to the original source resolution."""
        scale_factors = (
            self.overflow_buffered_source_extents / accumulator_shape[1:]
        )  # H, W
        return tuple(scale_factors)


class ScalarUniformExpansionPreprocessor(Preprocessor):
    """Preprocessor that expands scalar/vector values into uniform tiles using GCD compression."""

    def __init__(
        self, mask_tile_extents: np.ndarray, mask_tile_strides: np.ndarray
    ) -> None:
        self.compression_factors = np.gcd(mask_tile_strides, mask_tile_extents)
        self.adjusted_tile_extents = mask_tile_extents // self.compression_factors

    def reshape_storage(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (
            shape[0],
            *tuple(np.asarray(shape[1:], dtype=np.int64) // self.compression_factors),
        )

    def process(
        self, data_batch: np.ndarray, coords_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adjusted_tiles = np.zeros((*data_batch.shape, *self.adjusted_tile_extents))
        adjusted_tiles += data_batch[
            ..., *[np.newaxis] * len(self.adjusted_tile_extents)
        ]
        adjusted_coordinates = coords_batch // self.compression_factors[:, np.newaxis]
        return adjusted_tiles, adjusted_coordinates
