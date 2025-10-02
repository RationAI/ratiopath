# Adding Annotation Coverage to Tiles

This tutorial shows you how to enrich your tiling pipeline with annotation data—such as regions of interest, cancer boundaries, or other expert-marked areas. You’ll learn how to parse annotation files, compute coverage for each tile, and add this information to your dataset using `ratiopath`.

---

## Step 1: Understanding Annotation Coverage

Annotation coverage quantifies how much of a tile overlaps with annotated regions (e.g., cancerous tissue). This is useful for downstream analysis, training machine learning models, or filtering tiles based on biological relevance.

---

## Step 2: Parsing Annotation Files

To associate tiles with annotation data, you first need to parse the annotation files. `ratiopath` provides parsers for common formats such as ASAP XML and GeoJSON.

```python
from ratiopath.parsers import ASAPParser

annotation_path = row["path"].replace(".mrxs", ".xml")
parser = ASAPParser(annotation_path)
annotations = list(parser.get_polygons(name="...", part_of_group="..."))
```

- Replace `.mrxs` with your slide file extension as needed.
- Use appropriate parser for your annotation format.
- **Note:** The `name` and `part_of_group` arguments are regular expressions used to filter annotations based on their `Name` and `PartOfGroup` attributes.

---

## Step 3: Computing Tile Coverage

For each tile, define its region of interest (ROI) as a polygon. Then, calculate the fraction of the tile area covered by annotation polygons.
**Note:** In this example, the ROI is set to cover the entire tile, but you can use any geometry inside the tile as the ROI (e.g., a subregion, mask, or shape of interest). The ROI is always defined relative to the tile's coordinate system.


```python
from shapely import Polygon

roi = Polygon([
    (0, 0),
    (row["tile_extent_x"], 0),
    (row["tile_extent_x"], row["tile_extent_y"]),
    (0, row["tile_extent_y"]),
])
```

---

## Step 4: Attaching Coverage to Tile Metadata

Use the `tile_annotations` function to compute which annotation polygons overlap with the tile and return their intersection polygons. Add the coverage value to each tile row.

```python
from ratiopath.tiling import tile_annotations

def tiling_with_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotation_path = row["path"].replace(".mrxs", ".xml")
    parser = ASAPParser(annotation_path)
    annotations = list(parser.get_polygons(name="...", part_of_group="..."))

    roi = Polygon([
        (0, 0),
        (row["tile_extent_x"], 0),
        (row["tile_extent_x"], row["tile_extent_y"]),
        (0, row["tile_extent_y"]),
    ])

    coordinates = np.array(list(
        grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ))
    return [
        {
            "tile_x": coordinates[i, 0],
            "tile_y": coordinates[i, 1],
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "coverage": polygon.area / roi.area,
        }
        for i, polygon in enumerate(
            tile_annotations(
                annotations,
                roi,
                coordinates,
                row["downsample"],
            )
        )
    ]
```

- Each output row contains tile coordinates, metadata, and coverage value.

---

## Step 5: Integrating with the Pipeline

Apply the annotation coverage function to your tiles dataset using Ray Data’s `flat_map` or `map_batches`:

```python
tiles = slides.flat_map(tiling_with_annotations)
```

You can now filter, group, or analyze tiles based on their annotation coverage.

---

## Notes and Next Steps

- The `ASAPParser` can be replaced with other parsers (e.g., `GeoJSONParser`) depending on your annotation format.
- You can extend this approach to compute coverage for multiple annotation classes, or other metrics (e.g., distance to nearest annotation).

---

By adding annotation coverage, your pipeline produces richer tile metadata, enabling more targeted downstream workflows such as supervised learning, tissue quantification, or quality control.