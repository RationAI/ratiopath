# Attaching Overlay Data to Image Tiles

This tutorial demonstrates how to **enrich your primary tile dataset** with corresponding image patches (overlays) from a secondary **whole-slide image (WSI)**, such as a tissue mask or a heatmap. You will use **Ray Data's batch processing** with the `tile_overlay` and `tile_overlay_overlap` functions from the `ratiopath` library.

-----

## Step 1: Understanding Overlay Tiling and Coordinate System

Overlays (e.g., tissue masks) are often image files that are **aligned with the original WSI** but frequently have a **different physical resolution** ($\mu m/px$). The `ratiopath` functions, `tile_overlay` and `tile_overlay_overlap`, automatically handle these resolution differences and apply the necessary scaling to adjust tile coordinates before reading the overlay patches.

### **Essential Tile Overlay Metadata**

To correctly locate, scale, and extract the corresponding overlay patch, the processing function requires specific metadata for each tile, which must be present in the tile dataset:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `tile_x`, `tile_y` | `int` | Top-left pixel coordinates of the tile **in the primary slide's coordinate system**. |
| `tile_extent_x`, `tile_extent_y` | `int` | Width/Height of the tile **in the primary slide's coordinate system**. |
| `mpp_x`, `mpp_y` | `float` | (Optional) Physical resolution ($\mu m/px$) of the level used for the tile extraction. |
| `level` | `int` | (Optional) The pyramid level index used for the tile extraction. |
| `overlay_path` | `str` | The name of the column containing the file path to the overlay WSI. |

!!! Note
    Either `mpp_x` and `mpp_y`, or `level` must be provided to determine the resolution of the tile. Otherwise, the functions will raise a `ValueError`.

These columns must either be pre-computed and present in the tile dataset or generated dynamically using the `roi` function argument (see Step 5).

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

### 2.2 Applying the `tile_overlay` Processor

Instantiate `tile_overlay` to create a processor that reads the corresponding overlay patch (as a NumPy array) and attaches it to the input batch dictionary under the specified `store_key`.

```python
from ratiopath.tiling import tile_overlay

add_tissue_mask = tile_overlay(
    overlay_path_key="tissue_mask_path", # The column holding the overlay file path
    store_key="tissue_mask",             # The new column for the extracted patch array
)

# Apply the function to the Ray Dataset using map_batches
tiles_with_overlays = tiles.map_batches(add_tissue_mask)
```

The `tiles_with_overlays` dataset will now include a new column, `tissue_mask`, containing the raw overlay image patch (e.g., a NumPy array) corresponding to the tile.

-----

## Step 3: Computing Overlay Overlap/Coverage (`tile_overlay_overlap`)

Instead of retrieving the raw image patch, you can compute the **pixel count percentage** for every unique value in the resulting overlay patch, which is useful for filtering tiles based on content (e.g., how much tissue is present).

```python
from ratiopath.tiling import tile_overlay_overlap

add_tissue_mask_overlap = tile_overlay_overlap(
    overlay_path_key="tissue_mask_path",
    store_key="tissue_mask_overlap",
)

# Apply the function to the Ray Dataset using map_batches
tiles_with_overlap = tiles.map_batches(add_tissue_mask_overlap)
```

### Post-Processing the Overlap Data

The new column `tissue_mask_overlap` contains a dictionary for each tile, mapping **unique pixel values** (e.g., 0, 1, 255) in the overlay to their **area coverage** (as a fraction summing to 1).

You can easily post-process this to extract coverage of a specific class (e.g., the foreground tissue, often value `255`):

```python
def extract_foreground_coverage(tile: dict) -> dict:
    """Extracts the foreground coverage (value 255) from the overlap dictionary."""
    # Use .get(255, 0.0) to safely retrieve the value, defaulting to 0.0 if not present
    tile["tissue_coverage"] = tile["tissue_mask_overlap"].get(255, 0.0)
    return tile

tiles_with_tissue_coverage = tiles_with_overlap.map(extract_foreground_coverage)
```

-----

## Step 4: Defining a Region of Interest (ROI)

Both `tile_overlay` and `tile_overlay_overlap` accept an optional `roi` argument, which is a function that takes the tile's metadata and returns a modified set of coordinates and extents. This is crucial when the Region of Interest for the overlay is for instance a subregion **within the underlying tile's area**.

The `overlay` module provides a convenient `overlay_roi` factory function that adjusts the tile's coordinates and extents to define a specific ROI using **fractional offsets** and **extents** relative to the tile's full size.

### Example: Central 50% ROI

Here is an example of defining an ROI that extracts only the **central 50%** region of each tile:

```python
from ratiopath.tiling import overlay_roi

# Define an ROI that extracts the central 50% region of each tile
centered_roi = overlay_roi(
    offset_x_frac=0.25,  # Start reading at 25% of the tile width
    offset_y_frac=0.25,  # Start reading at 25% of the tile height
    extent_x_frac=0.5,   # Width of the ROI is 50% of the tile width
    extent_y_frac=0.5,   # Height of the ROI is 50% of the tile height
)

# You would then pass this to the tiling function:
# add_tissue_mask = tile_overlay(..., roi=centered_roi)
```

If no `roi` function is explicitly passed, it defaults to an identity function, which selects the **full tile area**.

> **Note:** The ROI is defined relative to the underlying tile in terms of fractional coordinates/extents. It is automatically scaled to the overlay's resolution when the patch is read.

-----

## Notes and Implementation Details

### Resolution Scaling ($\mu m/px$)

The module's core functionality relies on calculating the **scaling factor** ($S_x$, $S_y$) required to map the tile coordinates from the primary slide's resolution ($M_{slide}$) to the overlay's resolution ($M_{overlay}$):

$$S_{x} = \frac{M_{slide, x}}{M_{overlay, x}}$$

This factor is then applied to the tile's coordinates and dimensions. To ensure precise pixel addressing, the resulting floating-point values are rounded to the nearest integer:

$$\text{New\_Coordinate} = \lfloor (\text{Old\_Coordinate} \times S) + 0.5 \rfloor$$

### Backend Handling

The functions automatically detect the overlay file type (e.g., OpenSlide format or OME-TIFF) and use the appropriate backend (`_read_openslide_overlay` or `_read_tifffile_overlay`) for efficient access to the overlay image pyramid, abstracting the details from the user.