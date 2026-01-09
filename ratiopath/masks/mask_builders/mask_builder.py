import logging
import math
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
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


class NumpyMemMapMaskBuilderAllocatorMixin(MaskBuilder):
    """Mixin class to allocate accumulators as numpy memory-mapped files (memmaps).

    This mixin provides disk-backed storage for large masks that exceed available RAM.
    Memory mapping allows the OS to manage paging between disk and memory transparently,
    enabling processing of masks that would otherwise cause out-of-memory errors.

    **Temporary Files (default behavior when `filepath=None`):**
    A temporary file is created and used as backing storage. The file is deleted when
    the memmap is closed or garbage collected. Disk space is consumed during processing
    but automatically reclaimed afterward.

    **Explicit Files (when `filepath` is provided):**
    The memmap is backed by the specified file path, which persists after processing.
    This is useful for caching results or processing masks too large for temporary storage.
    If the file already exists, a FileExistsError is raised to prevent accidental data loss.

    This mixin uses NumPy's NPY format version 3.0 for compatibility with large arrays (>4GB).
    """

    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        filepath: Path | None = None,
        **kwargs: Any,
    ) -> np.memmap:
        if filepath is None:
            with tempfile.NamedTemporaryFile() as temp_file:
                return np.lib.format.open_memmap(
                    temp_file.name,
                    mode="w+",
                    shape=(channels, *mask_extents),
                    dtype=dtype,
                    version=(3, 0),
                )
        else:
            if filepath.exists():
                raise FileExistsError(f"Memmap filepath {filepath} already exists.")
            return np.lib.format.open_memmap(
                filepath,
                mode="w+",
                shape=(channels, *mask_extents),
                dtype=dtype,
                version=(3, 0),
            )


class NumpyArrayMaskBuilderAllocatorMixin(MaskBuilder):
    """Mixin class to allocate accumulators as numpy arrays.

    This mixin implements the `allocate_accumulator` method to create
    numpy arrays for the accumulator.
    """

    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.zeros((channels, *mask_extents), dtype=dtype)


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


class AveragingMaskBuilderMixin(MaskBuilder):
    """Mixin that implements averaging aggregation for overlapping tiles.

    This mixin accumulates tiles by addition and tracks the overlap count at each pixel.
    During finalization, the accumulated values are divided by the overlap count to compute
    the average value at each position. This is useful for:
    - Smoothly blending overlapping tile predictions
    - Reducing edge artifacts in sliding window processing
    - Computing ensemble averages from multiple passes

    The mixin allocates an additional `overlap_counter` accumulator with shape (1, *SpatialDims)
    to track how many tiles contributed to each pixel position.
    """

    overlap_counter: AccumulatorType

    def __init__(
        self, mask_extents: Int64[AccumulatorType, " N"], channels: int, **kwargs: Any
    ) -> None:
        super().__init__(mask_extents, channels, **kwargs)

    def setup_memory(self, mask_extents, channels, **kwargs) -> None:
        # Perform base allocation then allocate the overlap counter.
        super().setup_memory(mask_extents, channels, **kwargs)
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents, channels=1, **kwargs
        )

    def update_batch_impl(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        mask_tile_extents = np.asarray(data_batch.shape[2:])  # H, W, ...
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            self.accumulator[
                :,
                *acc_slices,
            ] += data
            self.overlap_counter[
                :,
                *acc_slices,
            ] += 1

    def finalize(self) -> tuple[AccumulatorType, ...]:
        # Average the accumulated mask by the overlap counts
        self.accumulator /= self.overlap_counter.clip(min=1)
        return self.accumulator, self.overlap_counter


class MaxMaskBuilderMixin(MaskBuilder):
    """Mixin that implements maximum aggregation for overlapping tiles.

    This mixin keeps only the maximum value at each pixel position when tiles overlap.
    No additional storage is required, and finalization is a no-op since the accumulator
    already contains the final max values. This is useful for:
    - Maximum intensity projection
    - Keeping the highest confidence prediction across overlapping tiles
    - Peak detection across multiple scales
    """

    def update_batch_impl(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        mask_tile_extents = np.asarray(data_batch.shape[2:], dtype=np.int64)  # H, W, ...
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,  # H, W, ...
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            self.accumulator[
                :,
                *acc_slices,
            ] = np.maximum(
                self.accumulator[
                    :,
                    *acc_slices,
                ],
                data,
            )


    def finalize(self) -> tuple[AccumulatorType]:
        return (self.accumulator,)


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


class AutoScalingScalarUniformValueConstantStrideMaskBuilder(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    ScalarUniformTiledMaskBuilder,
    AveragingMaskBuilderMixin
):
    """Mask builder combining auto-scaling with scalar uniform tiling.

    This builder automatically handles the complete pipeline for building masks from scalar/vector
    values when the network output resolution differs from input resolution:
    1. Computes output mask dimensions from source image and tile sizes
    2. Scales coordinates from input space to output space
    3. Compresses representation using GCD of tile extent/stride
    4. Expands scalar values into uniform tiles

    Typical use case: Neural network produces scalar predictions per input tile, and you want
    to create a coverage mask at a different resolution.

    Args:
        source_extents: Spatial dimensions of the source image (e.g., (10000, 10000)).
        channels: Number of channels in scalar predictions.
        source_tile_extents: Size of input tiles extracted from source (e.g., (512, 512)).
        source_tile_strides: Stride between input tiles in source space (e.g., (256, 256)).
        mask_tile_extents: Size you want each scalar to represent in output mask (e.g., (64, 64)).

    Example:
        ```python
        # Source image: 10000x10000
        # Input tiles: 512x512 at stride 256x256
        # Want each scalar to cover 64x64 in output
        builder = AutoScalingScalarUniformTiledMaskBuilder(
            source_extents=(10000, 10000),
            channels=3,
            source_tile_extents=(512, 512),
            source_tile_strides=(256, 256),
            mask_tile_extents=(64, 64),
        )
        # Output mask will be: (10000/512)*64 x (10000/512)*64 = 1250x1250
        # Coordinates should be provided in input space (0, 256, 512, ...)
        ```
    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            **kwargs,
        )


class AveragingScalarUniformTiledNumpyMaskBuilder(
    NumpyArrayMaskBuilderAllocatorMixin,
    ScalarUniformTiledMaskBuilder,
    AveragingMaskBuilderMixin,
):
    """Averaging scalar uniform tiled mask builder using numpy arrays as accumulators.

    The tiles are uniformly expanded from scalar/vector values before being assembled into the mask.
    The overlaps are averaged during finalization.
    The tiles are expected to have constant extents and strides.
    These can vary across dimensions (e.g. different width and height), but must not change across tiles.
    See `ScalarUniformTiledMaskBuilder` for details.

    Args:
        mask_extents (tuple[int, int]): Spatial dimensions of the mask to build.
        channels (int): Number of channels in the scalar values to be assembled into the mask.
        mask_tile_extents (tuple[int, int]): Extents of the tiles in mask space for each spatial dimension (height, width).
        mask_tile_strides (tuple[int, int]): Strides of the tiles in mask space for each spatial dimension (height, width).

    Example:
        ```python
        import openslide
        from rationai.masks.mask_builders import (
            AveragingScalarUniformTiledNumpyMaskBuilder,
        )
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.dimensions[LEVEL]
        vgg16_model = load_rationai_vgg16_model(...)  # load your pretrained model here
        mask_builder = AveragingScalarUniformTiledNumpyMaskBuilder(
            mask_extents=(slide_extent_y, slide_extent_x),
            channels=1,  # for binary classification
            mask_tile_extents=tile_extents,
            mask_tile_strides=tile_strides,
        )
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            features = vgg16_model.predict(tiles)  # features has shape (B, channels)
            mask_builder.update_batch(features, xs, ys)
        assembled_mask, overlap = mask_builder.finalize()
        plt.imshow(assembled_mask[0], cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()
        ```
    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            **kwargs,
        )


class MaxScalarUniformTiledNumpyMaskBuilder(
    NumpyArrayMaskBuilderAllocatorMixin,
    ScalarUniformTiledMaskBuilder,
    MaxMaskBuilderMixin,
):
    """Max scalar uniform tiled mask builder using numpy arrays as accumulators.

    The tiles are uniformly expanded from scalar/vector values before being assembled into the mask.
    The maximum value is taken at each pixel position during updates, no finalisation is required.
    The tiles are expected to have constant extents and strides.
    These can vary across dimensions (e.g. different width and height), but must not change across tiles.
    See `ScalarUniformTiledMaskBuilder` for details.

    Args:
        mask_extents (tuple[int, int]): Spatial dimensions of the mask to build.
        channels (int): Number of channels in the scalar values to be assembled into the mask.
        mask_tile_extents (tuple[int, int]): Extents of the tiles in mask space for each spatial dimension (height, width).
        mask_tile_strides (tuple[int, int]): Strides of the tiles in mask space for each spatial dimension (height, width).

    Example:
        ```python
        import openslide
        from rationai.masks.mask_builders import MaxScalarUniformTiledNumpyMaskBuilder
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.dimensions[LEVEL]
        vgg16_model = load_rationai_vgg16_model(...)  # load your pretrained model here
        mask_builder = MaxScalarUniformTiledNumpyMaskBuilder(
            mask_extents=(slide_extent_y, slide_extent_x),
            channels=1,  # for binary classification
            mask_tile_extents=tile_extents,
            mask_tile_strides=tile_strides,
        )
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            features = vgg16_model.predict(tiles)  # features has shape (B, channels)
            mask_builder.update_batch(features, xs, ys)
        (assembled_mask,) = mask_builder.finalize()
        plt.imshow(assembled_mask[0], cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()
        ```

    """

    def __init__(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        mask_tile_extents: Int64[AccumulatorType, " N"],
        mask_tile_strides: Int64[AccumulatorType, " N"],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            **kwargs,
        )


class AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
    NumpyMemMapMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilder2DMixin,
    AveragingMaskBuilderMixin,
):
    """Averaging mask builder with edge clipping using numpy memmaps as accumulators.

    This concrete builder combines three features:
    1. **Memory-mapped storage:** Handles large masks that exceed RAM capacity
    2. **Edge clipping:** Removes boundary artifacts from tiles
    3. **Averaging:** Smoothly blends overlapping tiles by computing pixel-wise averages

    The builder allocates two memmaps:
    - Main accumulator for tile data
    - Overlap counter (auto-suffixed with `.overlaps` if filepath is provided)

    Args:
        mask_extents: Spatial dimensions of the full-resolution mask (height, width).
        channels: Number of channels in the tiles/mask.
        clip: Edge clipping specification. Accepts:
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        filepath: Optional path for the main accumulator memmap. If None, uses temporary file.
            The overlap counter will use the same path with `.overlaps` suffix inserted before extension.

    Example:
        ```python
        import openslide
        from rationai.masks.mask_builders import AveragingClippingNumpyMemMapMaskBuilder
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.dimensions[LEVEL]
        vgg16_model = load_rationai_vgg16_model(...)  # load your pretrained model here
        mask_builder = AveragingClippingNumpyMemMapMaskBuilder(
            mask_extents=(slide_extent_y, slide_extent_x),
            channels=3,  # for RGB masks
            clip_top=4,
            clip_bottom=4,
            clip_left=4,
            clip_right=4,
        )
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            features = vgg16_model.predict(tiles)  # features has shape (B, channels)
            mask_builder.update_batch(features, xs, ys)
        (assembled_mask,) = mask_builder.finalize()
        plt.imshow(assembled_mask[0], cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()
        ```

    """

    def __init__(
        self,
        source_extents: Int64[AccumulatorType, " N"],
        source_tile_extents: Int64[AccumulatorType, " N"],
        source_tile_strides: Int64[AccumulatorType, " N"],
        mask_tile_extents: Int64[AccumulatorType, " N"],
        channels: int,
        clip: int | tuple[int, int] | tuple[int, int, int, int],
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            clip=clip,
            accumulator_filepath=accumulator_filepath,
            overlap_counter_filepath=overlap_counter_filepath,
        )

    def setup_memory(self, mask_extents, channels, accumulator_filepath=None, overlap_counter_filepath=None, **kwargs) -> None:
        self.accumulator = self.allocate_accumulator(
            mask_extents=mask_extents, channels=channels, filepath=accumulator_filepath
        )
        if overlap_counter_filepath is not None:
            counter_filepath = overlap_counter_filepath
        elif accumulator_filepath is not None:
            suffix = accumulator_filepath.suffix
            counter_filepath = accumulator_filepath.with_suffix(f".overlaps{suffix}")
        else:
            counter_filepath = None
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents, channels=1, filepath=counter_filepath
        )

    def get_vips_scale_factors(self) -> tuple[float, float]:
        """Get the scaling factors to convert the built mask back to the original source resolution.

        The ideas is to obtain coefficients for the pyvips.affine() function to rescale the assembled mask 
        back to the original source resolution after assembly and finalization. 
        Returns:
            tuple[float, float]: Scaling factors for height and width dimensions.
        """
        scale_factors = self.overflow_buffered_source_extents / self.accumulator.shape[1:]  # H, W
        return tuple(scale_factors)  # TODO: add tests for this method