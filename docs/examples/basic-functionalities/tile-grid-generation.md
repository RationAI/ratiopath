# Generate Tile Coordinates

!!! abstract "Overview"
    **Problem solved:** generate tile coordinates for dense tiling, overlapping tiling, or edge-aware extraction without reading slide pixels yet.

    **Use this example when:**

    - you want deterministic tile coordinates,
    - you need to control overlap with `stride`,
    - or you need explicit behavior for partial tiles at the slide boundary.

## Why This Approach

`grid_tiles` only generates coordinates.
That makes it the right tool when you want to inspect or filter the sampling plan before any expensive image IO happens.

## Example

```python
from ratiopath.tiling import grid_tiles

coordinates = list(
    grid_tiles(
        slide_extent=(8192, 6144),
        tile_extent=(1024, 1024),
        stride=(960, 960),
        last="keep",
    )
)

print(coordinates[:3])
```

??? example "Example output"
    ```text
    [array([0, 0]), array([960,   0]), array([1920,    0])]
    ```

??? info "Under the hood"
    `grid_tiles` does not inspect slide pixels.
    It only operates on three vectors: slide extent, tile extent, and stride.
    From those values it computes the maximum tile index per dimension and yields tile origins in row-major order.

    The `last` strategy changes how the upper edge of the coordinate range is computed:

    - `drop` uses floor division behavior and emits only tiles that fit cleanly.
    - `keep` uses ceiling behavior and may emit a final tile that extends beyond the nominal slide bounds.
    - `shift` also uses ceiling behavior, but clamps the final coordinates so the last tile stays inside the slide extent.

    Because this step is pure coordinate generation, it is cheap enough to run before any expensive backend reads.
    That is why it works well as the expansion step inside `flat_map`.

## Choosing `last`

Use `last="keep"` when:

- you want full grid coverage,
- and you can handle tiles that extend beyond the slide bounds downstream.

Use `last="drop"` when:

- you only want tiles that fit the slide extent cleanly.

Use `last="shift"` when:

- you want edge coverage,
- but still need each tile to stay inside the slide extent.

```python
edge_safe_coordinates = list(
    grid_tiles(
        slide_extent=(5000, 5000),
        tile_extent=(1024, 1024),
        stride=(1024, 1024),
        last="shift",
    )
)
```

??? info "Why this is useful in Ray pipelines"
    Separating coordinate generation from tile reading lets you keep the intermediate dataset small and structured.
    You can add slide IDs, annotation paths, cohort labels, or filtering rules while still working with scalar metadata.

    Only after repartitioning and pruning do you call `read_slide_tiles`.
    That typically improves throughput because the expensive image decode work happens on a smaller, better-balanced dataset.

## Turning Coordinates Into Rows

In a Ray workflow, you usually expand one slide row into many tile rows:

```python
from typing import Any

from ratiopath.tiling import grid_tiles


def expand_tiles(row: dict[str, Any]) -> list[dict[str, int | str]]:
    return [
        {
            "path": row["path"],
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
```

??? info "What the expansion step is doing"
    The expansion function copies slide-level fields such as `path`, `level`, and tile dimensions into every output row because downstream readers need those columns per tile.
    Conceptually, one slide row is being denormalized into many tile rows.

    That may look repetitive, but it fits Ray Data well because each tile becomes an independently schedulable record that can later be filtered, enriched, or joined with other tile-level signals.

## Related API

- [`ratiopath.tiling.tilers`](../../reference/tiling/tilers.md)
- [`ratiopath.tiling.read_slide_tiles`](../../reference/tiling/read_slide_tile.md)
