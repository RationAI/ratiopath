# Mask Builders

Mask builders are tools for assembling feature masks from neural network predictions or other tile-level data. They handle the complexity of combining overlapping tiles, scaling between coordinate spaces, and managing memory for large output masks.

## Overview

When processing whole-slide images with neural networks, you often need to:

1. Extract tiles from a slide
2. Run inference to get predictions or features for each tile
3. Assemble these predictions back into a full-resolution mask

Mask builders automate step 3, handling:

- **Coordinate transformation**: Converting from tile coordinates to mask coordinates
- **Overlap handling**: Averaging or taking the maximum when tiles overlap
- **Memory management**: Using in-memory arrays or memory-mapped files for large masks
- **Edge clipping**: Removing boundary artifacts from tiles

## MaskBuilderFactory

::: ratiopath.masks.mask_builders.MaskBuilderFactory

The factory composes a concrete mask builder dynamically based on the chosen capabilities.
All previous concrete builders are expressible as factory configurations.

## Example configurations

### Averaging scalar-expansion (numpy)

**Use case**: You have scalar predictions (e.g., class probabilities) for each tile and want to create a mask where each prediction is uniformly expanded to fill the tile's footprint. Overlapping regions are averaged.

**Example**: Creating a heatmap of tumor probability predictions.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import MaskBuilderFactory
import matplotlib.pyplot as plt

# Open slide and set up tiling parameters
LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]

# Load your model
vgg16_model = load_vgg16_model(...)  # load your pretrained model here

# Initialize mask builder
BuilderClass = MaskBuilderFactory.create(expand_scalars=True)
mask_builder = BuilderClass(
    mask_extents=(slide_extent_y, slide_extent_x),
    channels=1,  # for binary classification
    mask_tile_extents=tile_extents,
    mask_tile_strides=tile_strides,
)

# Process tiles
# Note: generate_tiles_from_slide is a placeholder - you must implement your own tile extraction logic
for tiles, xs, ys in generate_tiles_from_slide(
    slide, LEVEL, tile_extents, tile_strides, batch_size=32
):
    # tiles has shape (B, C, H, W)
    features = vgg16_model.predict(tiles)  # features has shape (B, channels)
    # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
    coords_batch = np.stack([ys, xs], axis=0)
    mask_builder.update_batch(features, coords_batch)

# Finalize and visualize
assembled_mask, overlap = mask_builder.finalize()
plt.imshow(assembled_mask[0], cmap="gray", interpolation="nearest")
plt.axis("off")
plt.show()
```

---

### Max scalar-expansion (numpy)

**Use case**: Similar to the averaging builder, but takes the maximum value at each pixel instead of averaging. Useful when you want to preserve the strongest signal.

**Example**: Creating activation maps from intermediate network layers.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import MaskBuilderFactory
import matplotlib.pyplot as plt
from rationai.explainability.model_probing import HookedModule

LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]

# Set up model with hooks to extract intermediate activations
vgg16_model = load_vgg16_model(...)
hooked_model = HookedModule(vgg16_model, layer_name="backbone.9")

BuilderClass = MaskBuilderFactory.create(
    aggregation="max",
    expand_scalars=True,
)
mask_builder = BuilderClass(
    mask_extents=(slide_extent_y, slide_extent_x),
    channels=1,
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

---

### Auto-scaling + clipping (memmap)

**Use case**: You have high-resolution feature maps from a network and need to:
- Handle masks too large for RAM (using memory-mapped files)
- Automatically scale coordinates from input to output space
- Remove edge artifacts from tiles

**Example**: Building attention maps with edge clipping to remove boundary artifacts.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import MaskBuilderFactory
from rationai.explainability.model_probing import HookedModule
import matplotlib.pyplot as plt

LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]

vgg16_model = load_vgg16_model(...)
hooked_model = HookedModule(vgg16_model, layer_name="backbone.9")

# This builder handles coordinate scaling and uses memory-mapped storage
BuilderClass = MaskBuilderFactory.create(
    use_memmap=True,
    auto_scale=True,
    clip=True,
)
mask_builder = BuilderClass(
    source_extents=(slide_extent_y, slide_extent_x),
    source_tile_extents=tile_extents,
    source_tile_strides=tile_strides,
    mask_tile_extents=(64, 64),  # output resolution per tile
    channels=3,  # for RGB masks
    px_to_clip=(4, 4, 4, 4),  # (top, bottom, left, right)
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

---

### Auto-scaling + scalar-expansion (numpy)

**Use case**: Your network outputs scalar predictions per tile, and you want each prediction to represent a fixed-size region in the output mask, automatically handling coordinate scaling.

**Example**: Creating a low-resolution classification map where each tile's prediction covers a 64×64 region.

```python
import numpy as np
import openslide
from ratiopath.masks.mask_builders import MaskBuilderFactory
import matplotlib.pyplot as plt

LEVEL = 3
tile_extents = (512, 512)
tile_strides = (256, 256)
slide = openslide.OpenSlide("path/to/slide.mrxs")
slide_extent_x, slide_extent_y = slide.level_dimensions[LEVEL]
classifier_model = load_classifier_model(...)

# Build a mask where each scalar prediction covers 64x64 pixels in output
BuilderClass = MaskBuilderFactory.create(
    auto_scale=True,
    expand_scalars=True,
)
mask_builder = BuilderClass(
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
    predictions = classifier_model.predict(tiles)  # predictions has shape (B, channels)
    # Stack ys and xs into coords_batch with shape (N, B) where N=2 (y, x dimensions)
    coords_batch = np.stack([ys, xs], axis=0)
    mask_builder.update_batch(predictions, coords_batch)

assembled_mask, overlap = mask_builder.finalize()
plt.imshow(assembled_mask[0], cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.show()
```

## Choosing a Mask Builder

| Configuration | Scalar/Feature Map | Aggregation | Memory | Auto-scaling | Edge Clipping |
|---------------|-------------------|-------------|---------|--------------|---------------|
| `expand_scalars=True` | Scalar | Average | RAM | No | No |
| `aggregation="max", expand_scalars=True` | Scalar | Max | RAM | No | No |
| `use_memmap=True, auto_scale=True, clip=True` | Feature Map | Average | Disk (memmap) | Yes | Yes |
| `auto_scale=True, expand_scalars=True` | Scalar | Average | RAM | Yes | No |

## Coordinate System Notes

All mask builders expect coordinates in the format `(N, B)` where:
- `N` is the number of spatial dimensions (typically 2 for height and width)
- `B` is the batch size

When implementing your own tile extraction logic (such as the `generate_tiles_from_slide` placeholder shown in examples), you should provide `xs` and `ys` arrays representing tile coordinates. Stack them as:

```python
coords_batch = np.stack([ys, xs], axis=0)  # Shape: (2, B)
```

Note the order: `[ys, xs]` not `[xs, ys]`, as the first dimension represents height (y) and the second represents width (x).
