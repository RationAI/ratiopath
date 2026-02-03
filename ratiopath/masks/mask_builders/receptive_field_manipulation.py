from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64, Shaped

from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilderABC


class EdgeClippingMaskBuilderMixin(MaskBuilderABC):
    """Mixin that clips edge pixels from tiles before accumulation.

    Edge clipping is useful for removing boundary artifacts from tiles, such as:
    - Zero-padding introduced by neural networks
    - Edge effects from convolution operations
    - Border artifacts from image processing

    This mixin intercepts `update_batch()`, clips the specified number of pixels from tile edges,
    adjusts the coordinates accordingly, and passes the clipped tiles to the next handler in the chain.

    **Important:** This mixin must appear BEFORE other mixins that implement `update_batch()` in the inheritance list (leftmost position) to ensure
    its `update_batch()` method is called before others in the MRO chain. Keep this in mind when composing new classes.

    **Usage:** Subclasses or derived classes must provide `clip_start_indices` and `clip_end_indices`
    during initialization to specify how many pixels to clip from the start (top/left) and
    end (bottom/right) edges in each spatial dimension.
    """

    clip_start_indices: Int64[AccumulatorType, " N"]
    clip_end_indices: Int64[AccumulatorType, " N"]

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        px_to_clip: int | tuple[int, ...],
        dtype: npt.DTypeLike,
        **kwargs: Any,
    ) -> None:
        """Initialize the edge clipping mixin.

        Args:
            mask_extents: Array of shape (N,) specifying the spatial dimensions of the mask to build.
            channels: Number of channels in the mask (e.g., 1 for grayscale, 3 for RGB, more for gathering intermediate CNN features).
            px_to_clip: Integer or tuple specifying pixels to clip from the edges of each dimension.
                1. If an integer is provided, it is applied uniformly to all edges in all dimensions.
                2. If a tuple of length N is provided, it specifies the number of pixels to clip from the start and end of each dimension.
                3. If a tuple of length 2N is provided, it specifies the number of pixels to clip from the start and end of each dimension separately, in the order: (dim1_start, dim1_end, dim2_start, dim2_end, ...).
            dtype: Data type for the accumulator.
            kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(mask_extents, channels, dtype=dtype, **kwargs)
        # first determine the number of spatial dimensions
        num_dims = len(mask_extents)

        # process px_to_clip into clip_start_indices and clip_end_indices
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

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        # Clip the edges of the input tiles
        extents = np.asarray(data_batch.shape[2:], dtype=np.int64)  # H, W, ...
        slices = tuple(
            slice(start, end)
            for start, end in zip(
                self.clip_start_indices,
                extents - self.clip_end_indices,
                strict=True,
            )
        )
        adjusted_coords_batch = coords_batch + self.clip_start_indices[:, np.newaxis]
        super().update_batch(
            data_batch=data_batch[..., *slices],  # type: ignore[arg-type]
            coords_batch=adjusted_coords_batch,
        )


class AutoScalingConstantStrideMixin(MaskBuilderABC):
    """Mixin that automatically scales coordinates to match input/output resolution differences.

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

    This mixin addresses this by calculating the scale factors from the input tile extents and output mask tile extents,
    adjusting the input coordinates accordingly before passing them to the next handler in the chain.
    This mixin also automatically computes the built mask spatial dimensions based on the tiled input total
    extent and the computed scale factors.

    Furthermore, this mixin readjusts the mask extent to cover for potential partial tiles at the edges.
    The safe extent is computed also for the source, so that the resulting mask can be properly rescaled back to the input resolution
    after assembly and finalization and then cropped to the original source extents to match exactly.
    The final rescaling and cropping is not part of this mixin and must be handled externally after mask finalization.
    Use the `overflow_buffered_source_extents` attribute to get the adjusted source extents including
    the overflow for partial tiles at the edges.

    This mixin works cooperatively with other mixins (like ScalarUniformTiledMaskBuilder) by computing
    the output space dimensions and passing them via kwargs, but keep in mind that those mixins must appear
    after this one in the inheritance list to ensure proper MRO.
    """

    overflow_buffered_source_extents: Int64[AccumulatorType, " N"]
    mask_tile_extents: Int64[AccumulatorType, " N"]
    source_tile_extents: Int64[AccumulatorType, " N"]

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike,
        **kwargs: Any,
    ) -> None:
        """Initialize the auto-scaling constant stride mixin.

        Here, the mask extents are computed automatically based on the source extents,
        source tile extents, source tile strides, and mask tile extents.

        Args:
            source_extents: Spatial dimensions of the entire input/source from which tiles are drawn.
            channels: Number of channels in the output.
            source_tile_extents: Spatial dimensions of the input/source tiles.
            mask_tile_extents: Spatial dimensions of the output tiles/mask.
            source_tile_strides: Stride between input/source tiles (optional, defaults to source_tile_extents).
            dtype: Data type for the accumulator.
            kwargs: Additional keyword arguments passed to the next class in MRO.
        """
        # self.source_extents = source_extents
        self.source_tile_extents = source_tile_extents
        self.mask_tile_extents = mask_tile_extents

        multiplied_ = source_tile_strides * self.mask_tile_extents
        # Ensure source_tile_strides * self.mask_tile_extents is divisible by self.source_tile_extents to avoid fractional strides
        if not np.all(multiplied_ % self.source_tile_extents == 0):
            raise ValueError(
                f"source_tile_strides * mask_tile_extents must be divisible by source_tile_extents in all dimensions,"
                f" but {source_tile_strides}*{self.mask_tile_extents}={multiplied_}, which is not divisible by {self.source_tile_extents}."
            )
        adjusted_mask_tile_strides = multiplied_ // self.source_tile_extents

        total_strides = (source_extents - source_tile_extents) / source_tile_strides
        total_strides = np.ceil(total_strides).astype(np.int64)
        # without the initial tile step, including partial tile at the edge
        self.overflow_buffered_source_extents = (
            total_strides * source_tile_strides
        ) + source_tile_extents
        overflow_buffered_mask_extents = (
            total_strides * adjusted_mask_tile_strides
        ) + self.mask_tile_extents

        # Call next in MRO with computed parameters
        super().__init__(
            mask_extents=overflow_buffered_mask_extents,
            channels=channels,
            mask_tile_extents=self.mask_tile_extents,
            mask_tile_strides=adjusted_mask_tile_strides,
            dtype=dtype,
            **kwargs,
        )

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        """Adjust input coordinates from source resolution to mask resolution before updating.

        This method rescales the input coordinates based on the ratio of source tile extents
        to mask tile extents to ensure proper alignment in the output mask space. This is handy
        when assembling outputs from models that change spatial resolution and alleviates the need for
        external coordinate transformations.
        """
        adjusted_coords_batch = (
            coords_batch * self.mask_tile_extents[:, np.newaxis]
        ) // self.source_tile_extents[:, np.newaxis]
        super().update_batch(data_batch=data_batch, coords_batch=adjusted_coords_batch)

    def get_vips_scale_factors(self) -> tuple[float, float]:
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
        scale_factors = (
            self.overflow_buffered_source_extents / self.accumulator.shape[1:]
        )  # H, W
        return tuple(scale_factors)


class ScalarUniformTiledMaskBuilder(MaskBuilderABC):
    """Mask builder that expands scalar/vector values into uniform tiles.

    This builder is designed for scenarios where each tile's content is uniform (constant value).
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

    This class can work cooperatively with other mixins (like AutoScalingConstantStrideMixin)
    that compute `mask_extents`. If `mask_extents` is already provided via kwargs (from a parent
    mixin), it will be used; otherwise it must be provided as a direct parameter.
    """

    compression_factors: Int64[AccumulatorType, " N"]
    adjusted_tile_extents: Int64[AccumulatorType, " N"]

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        dtype: npt.DTypeLike,
        **kwargs: Any,
    ) -> None:
        """Initialize the scalar uniform tiled mask builder.

        Args:
            mask_extents: Spatial dimensions of the mask to build (may be precomputed by a mixin).
            channels: Number of channels (features) in the scalar values.
            mask_tile_extents: Size of tiles in each dimension in mask space at the original resolution.
            mask_tile_strides: Stride between tile positions in mask space for each dimension.
            dtype: Data type for the accumulator.
            kwargs: Additional keyword arguments passed to the parent class.
        """
        self.compression_factors = np.gcd(mask_tile_strides, mask_tile_extents)
        adjusted_mask_extents = mask_extents // self.compression_factors
        self.adjusted_tile_extents = mask_tile_extents // self.compression_factors
        super().__init__(
            mask_extents=adjusted_mask_extents, channels=channels, dtype=dtype, **kwargs
        )

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        """For each scalar/vector in the batch, repeat it in each dimension to form a tile, then update the mask with the tile."""
        adjusted_tiles = np.zeros((*data_batch.shape, *self.adjusted_tile_extents))
        adjusted_tiles += data_batch[
            ..., *[np.newaxis] * len(self.adjusted_tile_extents)
        ]
        adjusted_coordinates = coords_batch // self.compression_factors[:, np.newaxis]
        super().update_batch(adjusted_tiles, coords_batch=adjusted_coordinates)
