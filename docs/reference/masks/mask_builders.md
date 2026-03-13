# Mask Builders

Mask builders are tools for assembling feature masks from neural network predictions or other tile-level data. They handle the complexity of combining overlapping tiles, scaling between coordinate spaces, and managing memory for large output masks using a flexible strategy-based architecture.

## Overview

When processing whole-slide images with neural networks, you often need to:

1. Extract tiles from a slide
2. Run inference to get predictions or features for each tile
3. Assemble these predictions back into a full-resolution mask

Mask builders automate step 3, handling:

- **Coordinate transformation**: Converting from tile coordinates to mask coordinates.
- **Overlap handling**: Averaging or taking the maximum when tiles overlap.
- **Memory management**: Using in-memory arrays or memory-mapped files for large masks.
- **Data transformation**: Clipping edges or expanding scalars into uniform tiles.

## MaskBuilder

::: ratiopath.masks.mask_builders.MaskBuilder

The `MaskBuilder` is the central orchestrator. It uses **composition** rather than inheritance to combine different behaviors. You configure it by providing:

- `storage`: Where the mask is stored (RAM or disk).
- `aggregation`: How overlapping tiles are merged (mean or max).
- `preprocessors`: A list of optional transformations applied to data and coordinates.

## Components

### Storage Strategies

::: ratiopath.masks.mask_builders.NumpyStorage
::: ratiopath.masks.mask_builders.MemMapStorage

### Aggregation Strategies

::: ratiopath.masks.mask_builders.MeanAggregator
::: ratiopath.masks.mask_builders.MaxAggregator

### Preprocessors

::: ratiopath.masks.mask_builders.AutoScalingPreprocessor
::: ratiopath.masks.mask_builders.EdgeClippingPreprocessor
::: ratiopath.masks.mask_builders.ScalarUniformExpansionPreprocessor

## Examples

### Averaging Scalar Expansion (NumPy)

**Use case**: You have scalar predictions (e.g., class probabilities) for each tile. Each prediction is uniformly expanded to fill the tile's footprint, and overlapping regions are averaged.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import (
    MaskBuilder, 
    ScalarUniformExpansionPreprocessor
)
import matplotlib.pyplot as plt

# Set up tiling parameters
LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_w, slide_h = slide.level_dimensions[LEVEL]

# Each scalar expands to the tile footprint. GCD compression is handled by the preprocessor.
scalar_prep = ScalarUniformExpansionPreprocessor(
    mask_tile_extents=np.array(tile_extents),
    mask_tile_strides=np.array(tile_strides)
)

mask_builder = MaskBuilder(
    shape=(1, slide_h, slide_w),
    storage="numpy",
    aggregation="mean",
    preprocessors=[scalar_prep],
    dtype=np.float32,
)

# Process tiles
for tiles, xs, ys in generate_tiles_from_slide(slide, LEVEL, tile_extents, tile_strides):
    features = model.predict(tiles)  # features shape: (B, 1)
    coords_batch = np.stack([ys, xs], axis=0) # shape: (2, B)
    mask_builder.update_batch(features, coords_batch)

# Finalize
results = mask_builder.finalize()
assembled_mask = results["mask"]
overlap_counter = results["overlap_counter"]

plt.imshow(assembled_mask[0], cmap="gray")
plt.show()
```

---

### Max Aggregation + Edge Clipping (MemMap)

**Use case**: You have high-resolution feature maps. You want to preserve the maximum signal where tiles overlap, remove 4 pixels from each tile edge to avoid artifacts, and use disk storage because the mask is very large.

```python
import numpy as np
from ratiopath.masks.mask_builders import (
    MaskBuilder,
    EdgeClippingPreprocessor
)

# Configure preprocessors
clip_prep = EdgeClippingPreprocessor(px_to_clip=4, num_dims=2)

mask_builder = MaskBuilder(
    shape=(3, 10000, 10000),
    storage="memmap",
    aggregation="max",
    preprocessors=[clip_prep],
    filename="large_mask.npy"
)

# ... updates ...

# MaxAggregator returns the NDArray directly
assembled_mask = mask_builder.finalize()
```

---

### Auto-scaling Coordinates

**Use case**: Your model's output tiles have different spatial dimensions than the input tiles (e.g., due to stride or pooling).

```python
from ratiopath.masks.mask_builders import MaskBuilder, AutoScalingPreprocessor

auto_scale = AutoScalingPreprocessor(
    source_extents=np.array([2000, 2000]),
    source_tile_extents=np.array([512, 512]),
    source_tile_strides=np.array([256, 256]),
    mask_tile_extents=np.array([128, 128])  # 4x downsampled output
)

mask_builder = MaskBuilder(
    shape=(1, *auto_scale.mask_extents),
    storage="numpy",
    aggregation="mean",
    preprocessors=[auto_scale]
)
```

## Coordinate System Notes

All mask builders expect coordinates in the format `(N, B)` where:
- `N` is the number of spatial dimensions (typically 2 for height and width).
- `B` is the batch size.

Note the order: `[ys, xs]` not `[xs, ys]`, as the first dimension represents height (y) and the second represents width (x), matching the NumPy `(C, H, W)` convention used by the builder.
