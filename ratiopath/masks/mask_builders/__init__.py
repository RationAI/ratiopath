from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.aggregation import (
    AveragingMaskBuilderMixin,
    MaxMaskBuilderMixin,
)
from ratiopath.masks.mask_builders.mask_builder import (
    AccumulatorType,
)
from ratiopath.masks.mask_builders.receptive_field_manipulation import (
    AutoScalingConstantStrideMixin,
    EdgeClippingMaskBuilder2DMixin,
    ScalarUniformTiledMaskBuilder,
)
from ratiopath.masks.mask_builders.storage import (
    NumpyArrayMaskBuilderAllocatorMixin,
    NumpyMemMapMaskBuilderAllocatorMixin,
)


__all__ = [
    "AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D",
    "AutoScalingScalarUniformValueConstantStrideMaskBuilder",
    "AveragingScalarUniformTiledNumpyMaskBuilder",
    "MaxScalarUniformTiledNumpyMaskBuilder",
]


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
        import numpy as np
        import openslide
        from ratiopath.masks.mask_builders import (
            AveragingScalarUniformTiledNumpyMaskBuilder,
        )
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]
        vgg16_model = load_vgg16_model(...)  # load your pretrained model here
        mask_builder = AveragingScalarUniformTiledNumpyMaskBuilder(
            mask_extents=(slide_extent_y, slide_extent_x),
            channels=1,  # for binary classification
            mask_tile_extents=tile_extents,
            mask_tile_strides=tile_strides,
        )
        # Note: generate_tiles_from_slide is a placeholder - you must implement your own tile extraction logic
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            features = vgg16_model.predict(tiles)  # features has shape (B, channels)
            # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
            coords_batch = np.stack([ys, xs], axis=0)
            mask_builder.update_batch(features, coords_batch)
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
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            dtype=dtype,
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
        import numpy as np
        import openslide
        from ratiopath.masks.mask_builders import MaxScalarUniformTiledNumpyMaskBuilder
        import matplotlib.pyplot as plt
        from rationai.explainability.model_probing import HookedModule

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]
        vgg16_model = load_vgg16_model(...)  # load your pretrained model here
        hooked_model = HookedModule(
            vgg16_model, layer_name="backbone.9"
        )  # example layer
        mask_builder = MaxScalarUniformTiledNumpyMaskBuilder(
            mask_extents=(slide_extent_y, slide_extent_x),
            channels=1,  # for binary classification
            mask_tile_extents=tile_extents,
            mask_tile_strides=tile_strides,
        )
        # Note: generate_tiles_from_slide is a placeholder - you must implement your own tile extraction logic
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            outputs = hooked_model.predict(tiles)  # outputs are not used directly
            features = hooked_model.get_activations("backbone.9")  # shape (B, C, H, W)
            # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
            coords_batch = np.stack([ys, xs], axis=0)
            mask_builder.update_batch(features, coords_batch)
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
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mask_extents=mask_extents,
            channels=channels,
            mask_tile_extents=mask_tile_extents,
            mask_tile_strides=mask_tile_strides,
            dtype=dtype,
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
    - Overlap counter (can be specified via overlap_counter_filepath, or auto-derived with `.overlaps` suffix if accumulator_filepath is provided)

    Args:
        source_extents: Spatial dimensions of the source space (height, width).
        source_tile_extents: Extents of tiles in the source space (height, width).
        source_tile_strides: Strides between tiles in the source space (height, width).
        mask_tile_extents: Extents of tiles in the mask space (height, width).
        channels: Number of channels in the tiles/mask.
        clip: Edge clipping specification. Accepts:
            - int: same clipping on all edges
            - (clip_y, clip_x): same for top/bottom and left/right
            - (clip_top, clip_bottom, clip_left, clip_right): individual edge control
        accumulator_filepath: Optional path for the main accumulator memmap. If None, uses temporary file.
        overlap_counter_filepath: Optional path for the overlap counter memmap. If None, derives from accumulator_filepath with `.overlaps` suffix.

    Example:
        ```python
        import numpy as np
        import openslide
        from ratiopath.masks.mask_builders import (
            AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D,
        )
        from rationai.explainability.model_probing import HookedModule
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]
        vgg16_model = load_vgg16_model(...)  # load your pretrained model here
        hooked_model = HookedModule(
            vgg16_model, layer_name="backbone.9"
        )  # example layer
        mask_builder = AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
            source_extents=(slide_extent_y, slide_extent_x),
            source_tile_extents=tile_extents,
            source_tile_strides=tile_strides,
            mask_tile_extents=(64, 64),  # output resolution per tile
            channels=3,  # for RGB masks
            clip=(4, 4, 4, 4),  # clip 4 pixels from each edge
        )
        # Note: generate_tiles_from_slide is a placeholder - you must implement your own tile extraction logic
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            output = vgg16_model.predict(tiles)  # outputs are not used directly
            features = hooked_model.get_activations("backbone.9")  # shape (B, C, H, W)
            # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
            coords_batch = np.stack([ys, xs], axis=0)
            mask_builder.update_batch(features, coords_batch)
        assembled_mask, overlap = mask_builder.finalize()
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
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
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
            dtype=dtype,
            **kwargs,
        )

    def setup_memory(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up memory for both the main accumulator and the overlap counter.

        This method does not call the super.setup_memory() method, breaking the MRO chain.
        This is intentional, as this mixin is specifically responsible for the allocation, handling both
        the main accumulator and the overlap counter, which requires special logic for filepaths and would
        be cumbersome to implement through multiple levels of inheritance.

        Args:
            mask_extents: Array of shape (N,) specifying the spatial dimensions of the mask to build.
            channels: Number of channels in the mask (e.g., 1 for grayscale, 3 for RGB).
            dtype: Data type for the accumulator (e.g., np.float32).
            accumulator_filepath: Optional Path to back the memmap file for the accumulator. If None, a temporary file is used.
            overlap_counter_filepath: Optional Path to back the memmap file for the overlap counter. If None, a temporary file is used.
            **kwargs: Additional keyword arguments for allocation.

        Returns:
            A numpy memmap array of shape (channels, *mask_extents) and specified dtype.
        """
        self.accumulator = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=channels,
            filepath=accumulator_filepath,
            dtype=dtype,
        )
        if overlap_counter_filepath is not None:
            counter_filepath = overlap_counter_filepath
        elif accumulator_filepath is not None:
            suffix = accumulator_filepath.suffix
            counter_filepath = accumulator_filepath.with_suffix(f".overlaps{suffix}")
        else:
            counter_filepath = None
        self.overlap_counter = self.allocate_accumulator(
            mask_extents=mask_extents,
            channels=1,
            filepath=counter_filepath,
            dtype=dtype,
        )

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
        return tuple(scale_factors)  # TODO: add tests for this method


class AutoScalingScalarUniformValueConstantStrideMaskBuilder(
    NumpyArrayMaskBuilderAllocatorMixin,
    AutoScalingConstantStrideMixin,
    ScalarUniformTiledMaskBuilder,
    AveragingMaskBuilderMixin,
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
        import numpy as np
        import openslide
        from ratiopath.masks.mask_builders import (
            AutoScalingScalarUniformValueConstantStrideMaskBuilder,
        )
        import matplotlib.pyplot as plt

        LEVEL = 3
        tile_extents = (512, 512)
        tile_strides = (256, 256)
        slide = openslide.OpenSlide("path/to/slide.mrxs")
        slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]
        classifier_model = load_classifier_model(...)  # load your pretrained model here

        # Build a mask where each scalar prediction covers 64x64 pixels in output
        mask_builder = AutoScalingScalarUniformValueConstantStrideMaskBuilder(
            source_extents=(slide_extent_y, slide_extent_x),
            source_tile_extents=tile_extents,
            source_tile_strides=tile_strides,
            mask_tile_extents=(64, 64),  # each scalar value expands to 64x64
            channels=3,  # for multi-class predictions
        )

        # Note: generate_tiles_from_slide is a placeholder - you must implement your own tile extraction logic
        for tiles, xs, ys in generate_tiles_from_slide(
            slide, LEVEL, tile_extents, tile_strides, batch_size=32
        ):
            # tiles has shape (B, C, H, W)
            predictions = classifier_model.predict(
                tiles
            )  # predictions has shape (B, channels)
            # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
            coords_batch = np.stack([ys, xs], axis=0)
            mask_builder.update_batch(predictions, coords_batch)

        assembled_mask, overlap = mask_builder.finalize()
        plt.imshow(assembled_mask[0], cmap="viridis", interpolation="nearest")
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
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_extents=source_extents,
            source_tile_extents=source_tile_extents,
            source_tile_strides=source_tile_strides,
            mask_tile_extents=mask_tile_extents,
            channels=channels,
            dtype=dtype,
            **kwargs,
        )
