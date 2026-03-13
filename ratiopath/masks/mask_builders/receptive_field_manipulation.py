from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    """Abstract base class for data and coordinate preprocessing steps."""

    def setup(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Optional setup method for preprocessors that require initialization with external parameters."""
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
    """Preprocessor that clips edge pixels from tiles before accumulation.

    Edge clipping is useful for removing boundary artifacts from tiles, such as:
    - Zero-padding introduced by neural networks
    - Edge effects from convolution operations
    - Border artifacts from image processing

    This preprocessor clips the specified number of pixels from tile edges,
    adjusts the coordinates accordingly, and passes the clipped tiles to the next handler in the chain.

    **Important:** This preprocessor should be called before other preprocessors.
    """

    def __init__(self, px_to_clip: int | tuple[int, ...]) -> None:
        """Initialize the edge clipping preprocessor.

        Args:
            px_to_clip: Integer or tuple specifying pixels to clip from the edges of each dimension.
                1. If an integer is provided, it is applied uniformly to all edges in all dimensions.
                2. If a tuple of length N is provided, it specifies the number of pixels to clip from the start and end of each dimension.
                3. If a tuple of length 2N is provided, it specifies the number of pixels to clip from the start and end of each dimension
                    separately, in the order: (dim1_start, dim1_end, dim2_start, dim2_end, ...).
        """
        self.px_to_clip = px_to_clip

    def setup(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        num_dims = len(shape) - 1  # Exclude channel dimension

        if isinstance(self.px_to_clip, int):
            clip_start_indices = (self.px_to_clip,) * num_dims
            clip_end_indices = (self.px_to_clip,) * num_dims
        elif isinstance(self.px_to_clip, tuple) and len(self.px_to_clip) == num_dims:
            clip_start_indices = self.px_to_clip
            clip_end_indices = self.px_to_clip
        elif (
            isinstance(self.px_to_clip, tuple) and len(self.px_to_clip) == 2 * num_dims
        ):
            clip_start_indices = self.px_to_clip[::2]
            clip_end_indices = self.px_to_clip[1::2]
        else:
            raise ValueError(
                f"px_to_clip must be an int, a tuple of {num_dims} ints, or a tuple of {2 * num_dims} ints."
            )

        self.clip_start_indices = np.asarray(clip_start_indices, dtype=np.int64)
        self.clip_end_indices = np.asarray(clip_end_indices, dtype=np.int64)
        return shape

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
    """Preprocessor that scales coordinates to match input/output resolution differences.

    Generally, neural networks such as CNNs and ViTs produce outputs at a different spatial resolution
    than their inputs due to pooling, striding, or patching operations. Assembling tile outputs of such models
    should happen on the output resolution grid, but input coordinates are often provided in the input resolution space.

    Since we want to retain the exact spatial correspondence between input tiles and output mask positions to be able to
    align predictions with the original input, we need to solve the resolution mismatch.

    Naively, one could rescale each mask tile using some kind of interpolation to match the input tile size and then assemble
    on the input resolution grid. However, this approach is computationally expensive and can introduce interpolation artifacts.
    Instead, this mixin assumes that the scaling between input and output resolutions is constant and can be expressed
    as a ratio of the input tile extent to the output tile extent in each dimension. It must also hold that the input tile stride
    multiplied by the output tile extent is divisible by the input tile extent to ensure integer strides in the mask coordinate space.

    This processor addresses this by calculating the scale factors from the input tile extents and output mask tile extents,
    adjusting the input coordinates accordingly before passing them to the next handler in the chain.
    This processor also automatically computes the built mask spatial dimensions based on the tiled input total
    extent and the computed scale factors.

    Furthermore, this processor readjusts the mask extent to cover for potential partial tiles at the edges.
    The safe extent is computed also for the source, so that the resulting mask can be properly rescaled back to the input resolution
    after assembly and finalization and then cropped to the original source extents to match exactly.
    The final rescaling and cropping is not part of this processor and must be handled externally after mask finalization.
    Use the `overflow_buffered_source_extents` attribute to get the adjusted source extents including
    the overflow for partial tiles at the edges.
    """

    def __init__(
        self,
        source_extents: np.ndarray,
        source_tile_extents: np.ndarray,
        source_tile_strides: np.ndarray,
        mask_tile_extents: np.ndarray,
    ) -> None:
        """Initialize the auto-scaling constant stride preprocessor.

        Here, the mask extents are computed automatically based on the source extents,
        source tile extents, source tile strides, and mask tile extents.

        Args:
            source_extents: Spatial dimensions of the entire input/source from which tiles are drawn.
            source_tile_extents: Spatial dimensions of the input/source tiles.
            source_tile_strides: Stride between input/source tiles (optional, defaults to source_tile_extents).
            mask_tile_extents: Spatial dimensions of the output tiles/mask.
        """
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
        """Adjust input coordinates from source resolution to mask resolution before updating.

        This method rescales the input coordinates based on the ratio of source tile extents
        to mask tile extents to ensure proper alignment in the output mask space. This is handy
        when assembling outputs from models that change spatial resolution and alleviates the need for
        external coordinate transformations.
        """
        adjusted_coords_batch = (
            coords_batch * self.mask_tile_extents[:, np.newaxis]
        ) // self.source_tile_extents[:, np.newaxis]
        return data_batch, adjusted_coords_batch

    def get_vips_scale_factors(
        self, accumulator_shape: tuple[int, ...]
    ) -> tuple[float, ...]:
        """Get the scaling factors to convert the built mask back to the original source resolution.

        The idea is to obtain coefficients for the pyvips.affine() function to rescale the assembled mask
        back to the original source resolution after assembly and finalization.
        To do that, we compute the ratio between the source extents and the final accumulator extents,
        taking into account any overflow buffering that was applied to the source extents, to maintain alignment.
        After the affine transformation, any extra pixels introduced by overflow buffering should be cropped out
        to the original source extents. The affine transformation nor the cropping are handled by the mask builder.

        The coefficients correspond to the height and width dimensions respectively.

        Returns:
            tuple[float, float]: Scaling factors for height and width dimensions.
        """
        return tuple(self.overflow_buffered_source_extents / accumulator_shape[1:])


class ScalarUniformExpansionPreprocessor(Preprocessor):
    """Preprocessor that expands scalar/vector values into uniform tiles using GCD compression.

    This preprocessor is designed for scenarios where each tile's content is uniform (constant value).
    Instead of storing full tiles, it compresses the representation by:
    1. Computing the GCD of tile extent and stride in each dimension
    2. Dividing the mask into a coarser grid with GCD granularity to save memory
    3. Expanding scalar values into the compressed grid

    This reduces memory usage and computation when tiles have uniform content, such as:
    - Per-tile classification scores
    - Per-tile feature vectors

    The builder automatically handles the coordinate transformation between the original
    and compressed grids.

    The resulting mask has spatial dimensions reduced by the GCD factors, but the effective resolution
    is as if the scalar values were expanded to full tiles defined in the `__init__` method.
    """

    def __init__(
        self, mask_tile_extents: np.ndarray, mask_tile_strides: np.ndarray
    ) -> None:
        """Initialize the scalar uniform tiled mask builder.

        Args:
            mask_tile_extents: Size of tiles in each dimension in mask space at the original resolution.
            mask_tile_strides: Stride between tile positions in mask space for each dimension.
        """
        self.compression_factors = np.gcd(mask_tile_strides, mask_tile_extents)
        self.adjusted_tile_extents = mask_tile_extents // self.compression_factors

    def setup(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (
            shape[0],
            *tuple(np.asarray(shape[1:], dtype=np.int64) // self.compression_factors),
        )

    def process(
        self, data_batch: np.ndarray, coords_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each scalar/vector in the batch, repeat it in each dimension to form a tile, then update the mask with the tile."""
        adjusted_tiles = np.zeros((*data_batch.shape, *self.adjusted_tile_extents))
        adjusted_tiles += data_batch[
            ..., *[np.newaxis] * len(self.adjusted_tile_extents)
        ]
        adjusted_coordinates = coords_batch // self.compression_factors[:, np.newaxis]
        return adjusted_tiles, adjusted_coordinates
