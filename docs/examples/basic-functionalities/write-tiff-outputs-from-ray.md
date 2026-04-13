# Write TIFF Outputs From Ray

!!! abstract "Overview"
    **Problem solved:** save NumPy image arrays from a Ray dataset as TIFF files without writing your own row-by-row export loop.

    **Use this example when:**

    - you already have image arrays in a Ray dataset,
    - you want one TIFF per row,
    - or you want to export masks, patches, or derived images for later inspection.

## Why This Approach

`VipsTiffDatasink` wraps `pyvips` inside a Ray datasink so dataset rows can be written directly to TIFF.
This is useful when your pipeline already lives in Ray and the natural output is a directory of images rather than another Parquet table.

## Example

```python
from ray.data.expressions import col

from ratiopath.ray import read_slides
from ratiopath.ray.datasource import VipsTiffDatasink
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ratiopath.tiling.utils import row_hash


def expand_tiles(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "path": row["path"],
            "slide_id": row["id"],
            "tile_x": x,
            "tile_y": y,
            "level": row["level"],
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


slides = read_slides("data", mpp=0.5, tile_extent=512, stride=512).map(row_hash)
tiles = slides.flat_map(expand_tiles).limit(32)

tiles_with_pixels = tiles.with_column(
    "tile",
    read_slide_tiles(
        col("path"),
        col("tile_x"),
        col("tile_y"),
        col("tile_extent_x"),
        col("tile_extent_y"),
        col("level"),
    ),
)

tiles_with_pixels.write_datasink(
    VipsTiffDatasink(
        path="tile_tiffs",
        data_column="tile",
        default_options={
            "compression": "lzw",
            "tile": True,
            "bigtiff": True,
        },
    )
)
```

??? info "Under the hood"
    `VipsTiffDatasink` expects a column containing NumPy arrays.
    For each row, it converts that array into a `pyvips.Image` and writes a TIFF buffer with the options you provide.

    This keeps export inside the distributed pipeline instead of forcing a manual collect-and-save step on the driver.

## When This Is Better Than Parquet

Use TIFF export when:

- another tool expects image files,
- you want to inspect outputs visually,
- or you are exporting masks or derived rasters rather than metadata.

Use Parquet when:

- the output is mostly scalar metadata,
- you want efficient downstream filtering and joins,
- or the image pixels are only an intermediate step.

## Related API

- [`ratiopath.ray.datasource.VipsTiffDatasink`](../../reference/ray/vips_tiff_datasink.md)
- [`ratiopath.tiling.read_slide_tiles`](../../reference/tiling/read_slide_tile.md)
