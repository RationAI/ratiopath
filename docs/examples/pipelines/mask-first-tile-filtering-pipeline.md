# Build A Mask-First Tile Filtering Pipeline

!!! abstract "Overview"
    **Problem solved:** generate tissue-mask overlays once, reuse them across distributed tiling runs, and keep the expensive RGB tile reads for only the tiles that pass mask-based filtering.

    **Use this pipeline when:**

    - you do not already have tissue-mask overlays,
    - you want reusable mask artifacts,
    - and background filtering should happen before large-scale RGB tile decoding.

## Workflow

1. Generate one tissue mask per slide.
2. Save the masks as aligned TIFF overlays.
3. Build the slide metadata table with `read_slides`.
4. Expand slides into tile coordinates with `grid_tiles`.
5. Attach mask overlap with `tile_overlay_overlap`.
6. Keep only tiles whose tissue fraction passes your threshold.
7. Optionally read RGB pixels only for the retained tiles.

## Example

```python
from pathlib import Path
from typing import Any

import pyvips
from ray.data.expressions import col
from shapely.geometry import box

from ratiopath.masks import tissue_mask
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles, tile_overlay_overlap
from ratiopath.tiling.utils import row_hash

mask_dir = Path("masks")
mask_dir.mkdir(exist_ok=True)
slide_paths = [
    Path("data/slide_a.svs"),
    Path("data/slide_b.ome.tif"),
]


def generate_mask(slide_path: str) -> str:
    slide = pyvips.Image.new_from_file(slide_path, access="sequential")
    mask, _ = tissue_mask(slide=slide, mpp=(8.0, 8.0))
    mask_path = mask_dir / f"{Path(slide_path).stem}_tissue_mask.tiff"
    mask.tiffsave(
        str(mask_path),
        tile=True,
        pyramid=True,
        bigtiff=True,
        compression="lzw",
    )
    return str(mask_path)


def add_mask_path(row: dict[str, Any]) -> dict[str, Any]:
    row["mask_path"] = str(mask_dir / f"{Path(row['path']).stem}_tissue_mask.tiff")
    return row


def expand_tiles(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "mask_path": row["mask_path"],
            "slide_id": row["id"],
            "tile_x": x,
            "tile_y": y,
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


for slide_path in slide_paths:
    generate_mask(str(slide_path))

slides = read_slides(
    [str(path) for path in slide_paths],
    mpp=0.5,
    tile_extent=512,
    stride=512,
)
slides = slides.map(row_hash).map(add_mask_path)

tiles = slides.flat_map(expand_tiles).repartition(target_num_rows_per_block=128)

roi = box(0, 0, 512, 512)

tiles = tiles.with_column(
    "tissue_overlap",
    tile_overlay_overlap(
        roi=roi,
        overlay_path=col("mask_path"),
        tile_x=col("tile_x"),
        tile_y=col("tile_y"),
        mpp_x=col("mpp_x"),
        mpp_y=col("mpp_y"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)

tiles = tiles.map(
    lambda row: {**row, "tissue_fraction": row["tissue_overlap"].get("255", 0.0)}
)
tiles = tiles.filter(lambda row: row["tissue_fraction"] >= 0.5)

tiles = tiles.with_column(
    "tile",
    read_slide_tiles(
        col("path"),
        col("tile_x"),
        col("tile_y"),
        col("tile_extent_x"),
        col("tile_extent_y"),
        col("level"),
    ),
    num_cpus=1,
    memory=4 * 1024**3,
)
```

Using one explicit `slide_paths` list keeps mask generation and slide ingestion aligned.
If you replace this with your own discovery logic, feed the same resolved paths into both stages.

??? info "Why this pipeline is efficient"
    The expensive mask generation happens once per slide and produces a reusable overlay.
    After that, later runs can decide which tiles to keep by querying the mask instead of decoding every RGB tile.

    This changes tissue filtering from a repeated image-read problem into a reusable metadata-enrichment problem.

## When To Use This Instead Of Pixel-Std Filtering

Use the mask-first path when:

- you will rerun tiling many times,
- you need a more stable tissue criterion than raw RGB variance,
- or the same masks will also be useful for QC or visualization.

Use the direct pixel-std filter when:

- you need a quick first pass,
- the dataset is small,
- or generating masks up front would be unnecessary overhead.

## Related API

- [`ratiopath.masks.tissue_mask`](../../reference/masks/tissue_mask.md)
- [`ratiopath.tiling.overlays`](../../reference/tiling/overlays.md)
- [`ratiopath.tiling.read_slide_tiles`](../../reference/tiling/read_slide_tile.md)
