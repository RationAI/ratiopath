from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilder


from typing import Any
from jaxtyping import Shaped, Int64
import numpy as np

class EdgeClippingMaskBuilderMixin(MaskBuilder):
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
        clip_start_indices: Int64[AccumulatorType, " N"],
        clip_end_indices: Int64[AccumulatorType, " N"],
        **kwargs: Any,
    ) -> None:
        super().__init__(mask_extents, channels, **kwargs)
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
        adjusterd_coords_batch = coords_batch + self.clip_start_indices[:, np.newaxis]
        super().update_batch(
            data_batch=data_batch[..., *slices],  # type: ignore[arg-type]
            coords_batch=adjusterd_coords_batch,
        )


class EdgeClippingMaskBuilder2DMixin(EdgeClippingMaskBuilderMixin):
    """2D-specific edge clipping mixin with convenient parameter formats.

    This specialized version of EdgeClippingMaskBuilderMixin provides intuitive parameter formats
    for 2D masks. The `clip` parameter accepts three formats:

    1. **Single integer:** `clip=5` → clips 5 pixels from all edges (top, bottom, left, right)
    2. **Tuple of 2 ints:** `clip=(5, 10)` → clips 5 pixels from top/bottom, 10 from left/right
    3. **Tuple of 4 ints:** `clip=(2, 3, 4, 5)` → clips 2 (top), 3 (bottom), 4 (left), 5 (right)

    **Important:** This mixin must appear FIRST in the inheritance list to ensure proper MRO.

    Example:
        ```python
        class MyBuilder(
            EdgeClippingMaskBuilder2DMixin,  # Must be first!
            NumpyArrayMaskBuilderAllocatorMixin,
            AveragingMaskBuilderMixin,
        ):
            pass


        builder = MyBuilder(mask_extents=(1024, 1024), channels=3, clip=8)
        ```
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        clip: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        **kwargs: Any,
    ) -> None:
        if isinstance(clip, int):
            clip_start_indices = clip_end_indices = (clip,) * len(mask_extents)
        elif isinstance(clip, tuple) and len(clip) == 2:
            clip_y, clip_x = clip
            clip_start_indices = (clip_y, clip_x)
            clip_end_indices = (clip_y, clip_x)
        elif isinstance(clip, tuple) and len(clip) == 4:
            clip_top, clip_bottom, clip_left, clip_right = clip
            clip_start_indices = (clip_top, clip_left)
            clip_end_indices = (clip_bottom, clip_right)
        else:
            raise ValueError(
                "clip must be an int, a tuple of two ints, or a tuple of four ints."
            )
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            clip_start_indices=np.asarray(clip_start_indices, dtype=np.int64),
            clip_end_indices=np.asarray(clip_end_indices, dtype=np.int64),
            **kwargs,
        )


class AutoScalingConstantStrideMixin(MaskBuilder):
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

    This mixin works cooperatively with other mixins (like ScalarUniformTiledMaskBuilder) by computing
    the output space dimensions and passing them via kwargs, but keep in mind that those mixins must appear
    after this one in the inheritance list to ensure proper MRO.

    Args:
        source_extents: Spatial dimensions of the entire input/source from which tiles are drawn.
        channels: Number of channels in the output.
        source_tile_extents: Spatial dimensions of the input/source tiles.
        mask_tile_extents: Spatial dimensions of the output tiles/mask.
        source_tile_strides: Stride between input/source tiles (optional, defaults to source_tile_extents).
    """

    # source_extents: Int64[AccumulatorType, " N"]
    overflow_buffered_source_extents: Int64[AccumulatorType, " N"]
    mask_extents: Int64[AccumulatorType, " N"]
    mask_tile_extents: Int64[AccumulatorType, " N"]
    source_tile_extents: Int64[AccumulatorType, " N"]

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        **kwargs: Any,
    ) -> None:
        # self.source_extents = source_extents
        self.source_tile_extents = source_tile_extents
        self.mask_tile_extents = mask_tile_extents

        multiplied_ = (source_tile_strides * self.mask_tile_extents)
        # Ensure source_tile_strides * self.mask_tile_extents is divisible by self.source_tile_extents to avoid fractional strides
        if not np.all(multiplied_ % self.source_tile_extents == 0):
            raise ValueError(
                f"source_tile_strides * mask_tile_extents must be divisible by source_tile_extents in all dimensions,"
                f" but {source_tile_strides}*{self.mask_tile_extents}={multiplied_}, which is not divisible by {self.source_tile_extents}."
            )
        adjusted_mask_tile_strides = multiplied_ // self.source_tile_extents

        # adjusted_mask_extents = (source_extents // self.source_tile_extents) * self.mask_tile_extents
        total_strides = (source_extents - source_tile_extents) / source_tile_strides
        total_strides = np.ceil(total_strides).astype(np.int64)
          # without the initial tile step, including partial tile at the edge
        self.overflow_buffered_source_extents = (total_strides * source_tile_strides) + source_tile_extents
        overflow_buffered_mask_extents = (total_strides * adjusted_mask_tile_strides) + self.mask_tile_extents


        # Call next in MRO with computed parameters
        super().__init__(
            mask_extents=overflow_buffered_mask_extents,
            channels=channels,
            mask_tile_extents=self.mask_tile_extents,
            mask_tile_strides=adjusted_mask_tile_strides,
            **kwargs,
        )

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        adjusted_coords_batch = ((coords_batch * self.mask_tile_extents[:, np.newaxis]) // self.source_tile_extents[:, np.newaxis])
        super().update_batch(
            data_batch=data_batch, coords_batch=adjusted_coords_batch
        )


class ScalarUniformTiledMaskBuilder(MaskBuilder):
    """Mask builder that expands scalar/vector values into uniform tiles.

    This builder is designed for scenarios where each tile's content is uniform (constant value).
    Instead of storing full tiles, it compresses the representation by:
    1. Computing the GCD of tile extent and stride in each dimension
    2. Dividing the mask into a coarser grid with GCD granularity
    3. Expanding scalar values into the compressed grid

    This reduces memory usage and computation when tiles have uniform content, such as:
    - Per-tile classification scores
    - Per-tile feature vectors

    The builder automatically handles the coordinate transformation between the original
    and compressed grids.

    This class can work cooperatively with other mixins (like AutoScalingConstantStrideMixin)
    that compute `mask_extents`. If `mask_extents` is already provided via kwargs (from a parent
    mixin), it will be used; otherwise it must be provided as a direct parameter.

    Args:
        mask_extents: Spatial dimensions of the mask to build (may be precomputed by a mixin).
        channels: Number of channels (features) in the scalar values.
        mask_tile_extents: Size of tiles in each dimension in mask space at the original resolution.
        mask_tile_strides: Stride between tile positions in mask space for each dimension.
    """

    compression_factors: Int64[AccumulatorType, " N"]
    adjusted_tile_extents: Int64[AccumulatorType, " N"]

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        **kwargs: Any,
    ) -> None:
        self.compression_factors = np.gcd(mask_tile_strides, mask_tile_extents)
        adjusted_mask_extents = mask_extents // self.compression_factors
        self.adjusted_tile_extents = mask_tile_extents // self.compression_factors
        super().__init__(
            mask_extents=adjusted_mask_extents, channels=channels, **kwargs
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