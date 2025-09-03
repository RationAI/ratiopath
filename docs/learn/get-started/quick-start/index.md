# Quick Start

Get from whole‑slide images to enriched tile metadata in minutes.

## Install
```bash
pip install "ratiopath"
```

## Minimal Pipeline
```python
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles

slides = read_slides("data", mpp=0.25, tile_extent=1024, stride=960)

def tiling(row):
    return [
        yield {
            "slide_id": row["id"],
            "tile_x": x,
            "tile_y": y,
            "level": row["level"],
        }
    for x, y in grid_tiles(
        (row["extent_x"], row["extent_y"]),
        (row["tile_extent_x"], row["tile_extent_y"]),
        (row["stride_x"], row["stride_y"]),
        last="keep",
    )
    ]
        

tiles = slides.flat_map(tiling)
tiles.show(5)
```

## Next Steps
- Build the full tiling pipeline: [Tiling Tutorial](./tiling.md)
- Add annotation coverage: [Annotation Coverage](./annotations.md)

## Tips
- Use .stats() after an action to inspect performance.
- Repartition before expensive pixel reads.
- Keep slide metadata separate; avoid duplication.

Ready? Jump to the [Tiling Tutorial](./tiling.md).
