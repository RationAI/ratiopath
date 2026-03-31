# Mask Builders

Mask builders are tools for assembling feature masks from neural network predictions or other tile-level data. They handle the complexity of combining overlapping tiles, scaling between coordinate spaces, and managing memory for large output masks using a flexible strategy-based architecture.

## Overview

When processing whole-slide images with neural networks, you often need to:

1. Extract tiles from a slide
2. Run inference to get predictions or features for each tile
3. Assemble these predictions back into a full-resolution mask

Mask builders automate step 3, handling:

- **Coordinate scaling**: Converting from source WSI coordinates to mask coordinates — including automatic GCD-based compression when tiles and strides share common factors.
- **Overlap handling**: Averaging or taking the maximum when tiles overlap.
- **Memory management**: Using in-memory arrays or memory-mapped files for large masks.
- **Scalar expansion**: Broadcasting scalar per-tile predictions `(B, C)` into spatial tiles automatically.
- **Edge clipping**: Removing border artifacts from model output tiles at update time.

## MaskBuilder

::: ratiopath.masks.mask_builders.MaskBuilder

The `MaskBuilder` is the central orchestrator. You configure it by providing:

- `source_extents`: Spatial dimensions of the source WSI (H, W, ...).
- `source_tile_extent`: Spatial dimensions of the model input tiles.
- `output_tile_extent`: Spatial dimensions of the model output tiles (can differ from input due to pooling/stride).
- `stride`: Stride between tiles in source resolution.
- `storage`: Where the mask is stored — `"inmemory"` (RAM) or `"memmap"` (disk-backed).
- `aggregation`: How overlapping tiles are merged — `MeanAggregator` (default) or `MaxAggregator`.

The mask shape is computed automatically from the source extents, tile extents, and stride using GCD-based compression for efficient memory use.

## Components

### Storage Strategies

::: ratiopath.masks.mask_builders.InMemory
::: ratiopath.masks.mask_builders.MemMap

### Aggregation Strategies

::: ratiopath.masks.mask_builders.MeanAggregator
::: ratiopath.masks.mask_builders.MaxAggregator

## Examples

### Averaging Scalar Predictions

**Use case**: You have scalar predictions (e.g., class probabilities) for each tile. Each prediction is uniformly expanded to fill the tile's footprint, and overlapping regions are averaged.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import MaskBuilder, MeanAggregator
import matplotlib.pyplot as plt

# Set up tiling parameters
LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_w, slide_h = slide.level_dimensions[LEVEL]

# output_tile_extent=(1, 1) means scalar data — the builder
# broadcasts (B, C) → (B, C, 1, 1) and upscales automatically.
mask_builder = MaskBuilder(
    source_extents=(slide_h, slide_w),
    source_tile_extent=tile_extents,
    output_tile_extent=(1, 1),
    stride=tile_strides,
    n_channels=1,
    storage="inmemory",
    aggregation=MeanAggregator,
    dtype=np.float32,
)

# Process tiles
for tiles, xs, ys in generate_tiles_from_slide(slide, LEVEL, tile_extents, tile_strides):
    features = model.predict(tiles)  # features shape: (B, 1)
    coords_batch = np.stack([ys, xs], axis=1)  # shape: (B, 2)
    mask_builder.update_batch(features, coords_batch)

# Finalize — MeanAggregator returns {"mask": ..., "overlap_counter": ...}
results = mask_builder.finalize()
assembled_mask = results["mask"]
overlap_counter = results["overlap_counter"]

plt.imshow(assembled_mask[0], cmap="gray")
plt.show()

# Always clean up to release storage resources
mask_builder.cleanup()
```

---

### Max Aggregation with Edge Clipping (MemMap)

**Use case**: You have high-resolution feature maps. You want to preserve the maximum signal where tiles overlap, remove border pixels from each tile edge to avoid artifacts, and use disk storage because the mask is very large.

```python
import numpy as np
from ratiopath.masks.mask_builders import MaskBuilder, MaxAggregator

# Dense output — output tiles match input tiles in spatial size
mask_builder = MaskBuilder(
    source_extents=(10000, 10000),
    source_tile_extent=(512, 512),
    output_tile_extent=(512, 512),
    stride=(256, 256),
    n_channels=3,
    storage="memmap",
    aggregation=MaxAggregator,
    dtype=np.float32,
    filename="large_mask.npy",  # persisted to disk
)

for tiles, coords in tile_generator:
    predictions = model.predict(tiles)  # (B, 3, 512, 512)
    # edge_clipping=4 removes 4px from each edge of every tile
    mask_builder.update_batch(predictions, coords, edge_clipping=4)

# MaxAggregator returns the accumulator NDArray directly
assembled_mask = mask_builder.finalize()
mask_builder.cleanup()
```

---

### Auto-Scaling Coordinates (Different Input/Output Resolution)

**Use case**: Your model's output tiles have different spatial dimensions than the input tiles (e.g., due to stride or pooling). The builder auto-scales coordinates between source and mask resolution.

```python
import numpy as np
from ratiopath.masks.mask_builders import MaskBuilder, MeanAggregator

# Model takes 512×512 input tiles, produces 128×128 output tiles (4× downsampled)
mask_builder = MaskBuilder(
    source_extents=(2000, 2000),
    source_tile_extent=(512, 512),
    output_tile_extent=(128, 128),
    stride=(256, 256),
    n_channels=1,
    storage="inmemory",
    aggregation=MeanAggregator,
    dtype=np.float32,
)

# Coordinates are always in SOURCE resolution — the builder
# handles the conversion to mask resolution internally.
for tiles, coords in tile_generator:
    predictions = model.predict(tiles)  # (B, 1, 128, 128)
    mask_builder.update_batch(predictions, coords)

results = mask_builder.finalize()
mask_builder.cleanup()
```

## Coordinate System Notes

All mask builders expect coordinates in the format `(B, N)` where:

- `B` is the batch size.
- `N` is the number of spatial dimensions (typically 2 for height and width).

Note the order: `[ys, xs]` not `[xs, ys]`, as the first dimension represents height (y) and the second represents width (x), matching the NumPy `(C, H, W)` convention used by the builder.

## Lifecycle

Always call `cleanup()` when you are done with a `MaskBuilder` to release storage resources (especially important for `MemMap` storage which holds file handles):

```python
mask_builder = MaskBuilder(...)
# ... update_batch calls ...
results = mask_builder.finalize()
mask_builder.cleanup()
```
