# Add Annotation And Overlay Signals

!!! abstract "Overview"
    **Problem solved:** attach annotation-derived geometry or overlay-derived coverage to tile metadata so you can filter or score tiles without moving full-resolution data through every step of the pipeline.

    **Use this example when:**

    - you have polygon annotations in ASAP, GeoJSON, or Darwin JSON format,
    - you have an aligned overlay such as a tissue mask or heatmap,
    - and you want tile-level metadata for downstream selection or supervision.

## Option 1: Add Annotation Coverage

Use `tile_annotations` when your source of truth is geometry.
This is a good fit for expert polygons, region-of-interest annotations, and class-specific boundaries.

```python
from typing import Any

import numpy as np
from shapely import Polygon

from ratiopath.parsers import ASAPParser
from ratiopath.tiling import grid_tiles, tile_annotations


def tiles_with_annotation_coverage(row: dict[str, Any]) -> list[dict[str, Any]]:
    parser = ASAPParser(row["annotation_path"])
    annotations = list(parser.get_polygons(name="Tumor.*"))

    roi = Polygon(
        [
            (0, 0),
            (row["tile_extent_x"], 0),
            (row["tile_extent_x"], row["tile_extent_y"]),
            (0, row["tile_extent_y"]),
        ]
    )

    coordinates = np.array(
        list(
            grid_tiles(
                slide_extent=(row["extent_x"], row["extent_y"]),
                tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
                stride=(row["stride_x"], row["stride_y"]),
                last="keep",
            )
        )
    )

    return [
        {
            "path": row["path"],
            "tile_x": coordinates[i, 0],
            "tile_y": coordinates[i, 1],
            "annotation_coverage": geometry.area / roi.area,
        }
        for i, geometry in enumerate(
            tile_annotations(
                annotations=annotations,
                roi=roi,
                coordinates=coordinates,
                downsample=row["downsample"],
            )
        )
    ]
```

??? info "Under the hood for annotation coverage"
    The parser converts file-format-specific annotations into Shapely geometries.
    `tile_annotations` then builds an `STRtree`, which is a spatial index used to avoid checking every annotation against every tile.

    For each tile coordinate, the function shifts the tile ROI into slide space, queries the tree for intersecting geometries, unions the intersections, and transforms the result back into tile-relative coordinates.
    That is why the returned geometry can be used directly to compute tile-local coverage such as `geometry.area / roi.area`.

    The `downsample` parameter matters because annotations are assumed to live at level 0.
    The ROI and coordinates often refer to a lower-resolution working level, so the function explicitly transforms between the two spaces.

## Option 2: Add Overlay Coverage

Use `tile_overlay_overlap` when your source of truth is an aligned raster overlay and you only need overlap fractions rather than the full overlay patch.
Start from a tile metadata table that already includes the overlay path for each slide:

```python
from typing import Any

from ray.data.expressions import col
from shapely.geometry import box

from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, tile_overlay_overlap


def expand_tiles(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "overlay_path": row["overlay_path"],
            "tile_x": x,
            "tile_y": y,
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


slides = read_slides("data", mpp=0.5, tile_extent=512, stride=512)
slides = slides.map(
    lambda row: {
        **row,
        "overlay_path": row["path"].replace(".svs", "_tissue_mask.tiff"),
    }
)
tiles = slides.flat_map(expand_tiles)

roi = box(0, 0, 512, 512)

tiles_with_overlap = tiles.with_column(
    "tissue_overlap",
    tile_overlay_overlap(
        roi=roi,
        overlay_path=col("overlay_path"),
        tile_x=col("tile_x"),
        tile_y=col("tile_y"),
        mpp_x=col("mpp_x"),
        mpp_y=col("mpp_y"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)
```

If your overlay files follow a different naming convention, replace the `overlay_path` mapping with your own lookup.

## Option 3: Attach The Aligned Overlay Patch

Use `tile_overlay` when you need the overlay pixels themselves, for example to feed a model, inspect a mask visually, or compute custom patch-level statistics that are not expressible as simple per-value fractions.
Reusing the `tiles` dataset from the previous setup:

```python
from ray.data.expressions import col
from shapely.geometry import box

from ratiopath.tiling import tile_overlay

roi = box(128, 128, 384, 384)

tiles_with_overlay = tiles.with_column(
    "tissue_overlay",
    tile_overlay(
        roi=roi,
        overlay_path=col("overlay_path"),
        tile_x=col("tile_x"),
        tile_y=col("tile_y"),
        mpp_x=col("mpp_x"),
        mpp_y=col("mpp_y"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)
```

The resulting column stores two arrays per row:

- the overlay data itself,
- and a mask whose `True` values mark pixels outside the ROI polygon.

??? info "Under the hood for overlay coverage"
    Overlay processing is different from annotation processing because the source is raster data, not geometry.
    `tile_overlay_overlap` first resolves the overlay pyramid level closest to the tile resolution, scales the tile coordinates into that overlay space, and reads the corresponding overlay region.

    If you provide an arbitrary ROI, the implementation reads the ROI bounding box and then masks out pixels outside the ROI polygon.
    The overlap function then counts the unique unmasked values and returns normalized fractions per value.

    This is cheaper than storing full overlay patches when all you need is tile-level summary metadata such as tissue fraction or class proportion.

## How To Choose

Use annotations when:

- labels are authored as polygons or points,
- you need geometry-level control,
- or you want parser-based filtering before coverage is computed.

Use overlays when:

- labels already exist as aligned raster masks,
- you need fast per-tile value fractions,
- or you want the option to attach the overlay patch itself later.

Use `tile_overlay` instead of `tile_overlay_overlap` when:

- downstream code needs the raster patch itself,
- you want to run custom NumPy logic over the masked overlay,
- or you want to inspect or save the aligned overlay data for debugging.

??? info "Geometry-first vs raster-first reasoning"
    Choose the geometry path when labels are semantically authored objects and you need exact polygon behavior, class filtering, or later geometric operations.
    Choose the raster path when labels are already materialized as aligned masks or heatmaps and your downstream logic depends on pixel fractions rather than on polygon topology.

    The two paths solve similar problems but operate in different representations.
    Keeping that distinction clear helps avoid unnecessary format conversions and makes the pipeline easier to reason about.

## Related API

- [`ratiopath.parsers.ASAPParser`](../../reference/parsers/asap.md)
- [`ratiopath.parsers.GeoJSONParser`](../../reference/parsers/geojson.md)
- [`ratiopath.parsers.Darwin7JSONParser`](../../reference/parsers/darwin.md)
- [`ratiopath.tiling.annotations`](../../reference/tiling/annotations.md)
- [`ratiopath.tiling.overlays`](../../reference/tiling/overlays.md)
