# Attaching Overlay Data to Image Tiles

This tutorial demonstrates how to **enrich your primary tile dataset** with corresponding image patches (overlays) from a secondary **whole-slide image (WSI)**, such as a tissue mask or a heatmap. You will use **Ray Data's batch processing** with the `tile_overlay` and `tile_overlay_overlap` functions from the `ratiopath` library.

-----

## Step 1: Understanding Overlay Tiling and Coordinate System

Overlays (e.g., tissue masks) are often image files that are **aligned with the original WSI** but frequently have a **different physical resolution** ($\mu m/px$). The `ratiopath` functions, `tile_overlay` and `tile_overlay_overlap`, automatically handle these resolution differences and apply the necessary scaling to adjust tile coordinates before reading the overlay patches.

### **Essential Tile Overlay Metadata**

Both presented functions, `tile_overlay` and `tile_overlay_overlap`, are Ray's User Defined Function Expressions (`UDFExpr`) that process batches of data provided as column expressions. Both functions further require a Region of Interest (ROI) which defines the area of interets relative to each provided underlaying tile. The ROI must be define in the underlying tile's space (physical resolution). There are no limits for the shape of the ROI it can be an arbitrary polygon, neither the size, it may exceed the underlying tile. The ROIs that may exceed beiond the overlay image bounds are automatically clipped to the overlay image bounds.

To correctly locate, scale, and extract the corresponding overlay patch, the processing function require following metadata.

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `tile_x`, `tile_y` | `int` | Top-left pixel coordinates of the tile **in the primary slide's coordinate system**. |
| `tile_extent_x`, `tile_extent_y` | `int` | Width/Height of the tile **in the primary slide's coordinate system**. |
| `mpp_x`, `mpp_y` | `float` | Physical resolution ($\mu m/px$) of the level used for the tile extraction. |
| `overlay_path` | `str` | File path to the overlay WSI. |

-----

## Step 2: Attaching the Overlay Patches (`tile_overlay`)

First, assume you have prepared a Ray Dataset of tiles (`tiles`) following a quick-start guide, where each record contains a `path` column (the primary WSI file path).

### 2.1 Augmenting with Overlay Path

You must augment the tile dataset to include a column with the file path to the **overlay WSI** that corresponds to each tile.

```python
# Follow the tutorial in Tiling Quick Start to prepare the Ray Dataset of tiles
slides = ...
tiles = ...

def add_overlay_path(batch: dict) -> dict:
    """Adds the overlay path for each tile in the batch."""
    # Example: Replace the WSI extension with the mask file extension
    batch["tissue_mask_path"] = batch["path"].str.replace(".mrxs", "_tissue_mask.tiff")
    return batch

tiles = tiles.map_batches(add_overlay_path)
```

### 2.2 Define the Region of Interest (ROI)

For this example, we will use a simple rectangular ROI that corresponds to the center view of each tile with half the width and height.

```python
from shapely.geometry import box


# Assuming the tile size is 512x512 pixels
roi = box(128, 128, 384, 384) # A box from (128,128) to (384,384)
```

### 2.3 Extracting Overlay Patches

We can now use the `tile_overlay` function to extract the overlay patches corresponding to each tile. We will store the extracted overlay patches in a new column called `tissue_overlay`.

````python
from ratiopath.tiling import tile_overlay


tile_with_overlay = tiles.add_column(
    "tissue_overlay",  # New column name for the overlay patch
    tile_overlay(
        roi=roi,
        overlay_path=col("tissue_mask_path"),
        tile_x=col("tile_x"),
        tile_y=col("tile_y"),
        mpp_x=col("mpp_x"),
        mpp_y=col("mpp_y"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)
````

The `tiles_with_overlays` dataset will now include a new column, `tissue_overlay`, containing the raw overlay image patches corresponding to each tile. For optimization purposes, the overlay patches are of the shape of the bounding box of the provided ROI. Unfortunately, at the moment we cannot use masked arrays directly in Ray Dataset. So instead of a numpy masked array, we provide the data and the mask as 2 separate arrays. The mask indicates which pixels are inside the ROI and which are outside.

-----

## Step 3: Computing Overlay Overlap/Coverage (`tile_overlay_overlap`)

Instead of retrieving the raw image patch, you can compute the **pixel ratio** for every unique value in the resulting overlay patch, which is useful for filtering tiles based on content (e.g., how much tissue is present). The setup is similar to the previous step.

```python
from ratiopath.tiling import tile_overlay_overlap


tissue_tiles = tiles.add_column(
    "tissue_overlap",  # New column name for the overlay patch
    tile_overlay_overlap(
        roi=roi,
        overlay_path=col("tissue_mask_path"),
        tile_x=col("tile_x"),
        tile_y=col("tile_y"),
        mpp_x=col("mpp_x"),
        mpp_y=col("mpp_y"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)
```

### Post-Processing the Overlap Data

The new column `tissue_mask_overlap` contains a dictionary for each tile, mapping **unique pixel values** (e.g., 0, 1, 255) in the overlay to their **area coverage** (as a fraction summing to 1). Due to PyArrow's dictionary limitations, which is used by Ray for storing the data, the keys are strings and the keys are shared across all rows. Missing keys are filled with `None`.
You can easily post-process this to extract coverage of a specific class (e.g., the foreground tissue, often value `255`):

```python
def extract_foreground_coverage(tile: dict) -> dict:
    """Extracts the foreground coverage (value 255) from the overlap dictionary."""
    # Use .get(255, 0.0) to safely retrieve the value, defaulting to 0.0 if not present
    tile["tissue_coverage"] = tile["tissue_mask_overlap"].get('255', 0.0)
    return tile

tiles_with_tissue_coverage = tiles_with_overlap.map(extract_foreground_coverage).filter(
    lambda tile: tile["tissue_coverage"] >= 0.5  # Keep tiles with at least 50% tissue
)
```

-----

## Notes and Implementation Details

### Resolution Scaling ($\mu m/px$)

The module's core functionality relies on calculating the **scaling factor** ($S_x$, $S_y$) required to map the tile coordinates from the primary slide's resolution ($M_{slide}$) to the overlay's resolution ($M_{overlay}$):

$$S_{x} = \frac{M_{slide, x}}{M_{overlay, x}}$$

This factor is then applied to the tile's coordinates and dimensions. To ensure precise pixel addressing, the resulting floating-point values are rounded to the nearest integer:

$$\text{New\_Coordinate} = \lfloor (\text{Old\_Coordinate} \times S) + 0.5 \rfloor$$

### Backend Handling

The functions automatically detect the overlay file type (e.g., OpenSlide format or OME-TIFF) and use the appropriate backend (`_read_openslide_overlay` or `_read_tifffile_overlay`) for efficient access to the overlay image pyramid, abstracting the details from the user.