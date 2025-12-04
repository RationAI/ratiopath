# Building a Tiling Pipeline

You will build a simple but efficient and scalable tiling pipeline for histopathology slides.
This tutorial does not assume any existing knowledge of parallel processing frameworks.
We will use [Ray Data](https://docs.ray.io/en/latest/data/data.html) and explain the necessary concepts as we go.
The techniques you'll learn are fundamental to building scalable data processing workflows with `ratiopath`.

This tutorial is divided into several sections:

-   [Setup](#setup) will give you a starting point to follow the tutorial.
-   [Overview](#overview) will teach you the fundamentals of processing slides with `ratiopath` and Ray.
-   [Building the Pipeline](#building-the-pipeline) will guide you through the most common techniques in a tiling workflow.


### What are you building?

In this tutorial, you'll build a pipeline that reads whole-slide images, generates a grid of tiles, filters out background tiles, and saves the results as a Parquet file.

You can see what it will look like when you’re finished here:

```python
from typing import Any

from ray.data.expressions import col

from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ratiopath.tiling.utils import row_hash


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
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


if __name__ == "__main__":
    slides = read_slides("data", mpp=0.25, tile_extent=1024, stride=1024 - 64)

    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides.write_parquet("slides")

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=128
    )

    tissue_tiles = tiles.with_column(
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
    ).filter(lambda row: row["tile"].std() > 8)

    tissue_tiles = tissue_tiles.drop_columns(
        ["tile", "path", "level", "tile_extent_x", "tile_extent_y"]
    )

    tissue_tiles.write_parquet("tiles")
```

If the code doesn't make sense to you yet, don't worry!
We’ll break it down and reconstruct it piece by piece.


## Setup

Before you start, make sure you have `ratiopath` installed.
You will also need a directory with some sample whole-slide images (`.svs`, `.tif`, `.ndpi`, `.ome.tif`, ...).
For this tutorial, we'll assume they are in a folder named `data/`.


## Overview

Now that you're set up, let's get an overview of ratiopath!

### Processing Slides as a Table

Instead of "open this giant image and loop over pixels," we're going to treat our collection of slides as a table of metadata. Each row represents a slide, with columns for its path, dimensions, resolution, and the tile size you want to use.

Why care? Because metadata is tiny. Moving metadata through the cluster is cheap; reading gigapixel tiles is not. We defer real I/O until *just before* we need pixels.


### Parallel Processing with Ray Data

Ray Data is a library for scaling data processing. It takes your table of slides, splits it into parallel blocks, and runs your processing steps on multiple CPU cores.

![Ray data blocks](https://docs.ray.io/en/latest/_images/dataset-arch.svg){ align=center }

You just need to define the work to be done on each row (or a batch of rows), and Ray handles the rest. We'll use a few key methods:

-   [`map()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html): Apply a function to each row.
-   [`flat_map()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.flat_map.html): Apply a function that can return multiple output rows for each input row.
-   [`filter()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.filter.html): Remove rows based on a condition.
-   [`map_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html): Apply a function to a batch of rows at once.
-   [`with_column()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.with_column.html): Add a new column to the dataset.


!!! note "Lazy Execution"
    When you call a method like `map()` or `filter()`, Ray doesn't actually *do* anything yet. It just builds an internal *logical plan*—a recipe of what you want to do (`Read -> Map -> FlatMap...`). The work only starts when you call an *action* like `write_parquet()` or `count()`. This "lazy" approach allows Ray to optimize the entire workflow before running it. You can even inspect the plan and performance stats with `dataset.stats()` after an action.


## Building the Pipeline

Let's get started.
This is where you'll spend the rest of the tutorial.

### Step 1: Reading Slide Metadata

First, you'll create a `Dataset` from your slides. A `Dataset` is just Ray's name for a table that can be processed in parallel. Each row in our `Dataset` will correspond to a single slide.

`ratiopath` provides a custom `read_slides` method that plugs directly into Ray. You tell it where your slides are (`"data"`), and what resolution you want to work with. Here, we're asking for a resolution where one pixel is about `0.25` micrometers (`mpp=0.25`). The datasource will automatically find the best magnification level in each slide file to match this.

You also provide `tile_extent` (the size of your tiles, e.g., 1024x1024 pixels) and `stride` (how far to move before starting the next tile). These are added as metadata to each row, ready for the tiling step later.

```python
from ratiopath.ray import read_slides

slides = read_slides("data", mpp=0.25, tile_extent=1024, stride=1024 - 64)
```

This returns a `Dataset` where each row is a dictionary holding the metadata for one slide. It looks something like this:

```json
{
  "path": "/abs/path/slide1.svs",
  "extent_x": 84320,
  "extent_y": 61120,
  "tile_extent_x": 1024,
  "tile_extent_y": 1024,
  "stride_x": 960,
  "stride_y": 960,
  "mpp_x": 0.25,
  "mpp_y": 0.25,
  "level": 2,
  "downsample": 4.0
}
```

### Step 2: Creating Unique Slide IDs

Next, you'll give each slide a unique ID. Why? If you process the same slide with different parameters (like different tile sizes), you'll want a consistent way to identify which slide a tile came from. We'll do this by hashing the slide's metadata.

You'll use `.map()` to apply the `row_hash` function to every row in your `Dataset`. Then, you'll save this slide-level table to a Parquet file. This is a good practice because you want to store this metadata only once for all the tiles in a slide.


```python
from ratiopath.tiling.utils import row_hash

slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
slides.write_parquet("slides")
```

!!! note "Our First Action!"
    `write_parquet()` is our first *action*. Only now does Ray actually execute the plan (`Read -> Map`). Before this call, nothing had run.


### Step 3: Expand Slides into Tile Coordinates

Now it's time to generate the grid of tiles for each slide. You'll use `flat_map()` because each slide row will "explode" into many tile rows.

First, define a `tiling` function. This function takes a slide row (which contains all the metadata) and uses `grid_tiles` to generate a list of `(x, y)` coordinates for the top-left corner of each tile. Each coordinate becomes a new row, and we also copy relevant parent slide's metadata into it.

```python
from typing import Any
from ratiopath.tiling import grid_tiles

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
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

tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2)
```

After `flat_map`, all tiles from a single slide are likely in the same data block. To get better parallelism for the next, more intensive step, you should `repartition` the dataset. This shuffles the rows (our tiles) so they are more evenly distributed across many smaller blocks, allowing Ray to spread the work across more CPU cores.

```python
tiles = tiles.repartition(target_num_rows_per_block=128)
```

!!! question "Choosing target_num_rows_per_block"
    Aim for blocks large enough to amortize scheduling overhead (hundreds–thousands of rows) but small enough to balance across cores and fit in memory when you later attach pixel arrays.



### Step 4: Reading and Filtering Tiles

So far, you've only worked with coordinates.
Now, you'll read the actual image data for each tile and filter out the ones that don't contain tissue.

The `read_slide_tiles` function reads the corresponding tile regions from the original slide file and adds it to the dataset as a NumPy array. The function is designed to work efficiently with batches of tiles. The inputs are columns from the `tiles` dataset, specified using Ray's column expressions ([`col("column_name")`](https://docs.ray.io/en/latest/data/api/doc/ray.data.expressions.col.html#ray.data.expressions.col)). 

```python
from ray.data.expressions import col

from ratiopath.tiling import read_slide_tiles


tiles_with_pixels = tiles.with_column(
    "tile",  # Name of the new column to add.
    read_slide_tiles(
        col("path"),
        col("tile_x"),
        col("tile_y"),
        col("tile_extent_x"),
        col("tile_extent_y"),
        col("level"),
    ),
    num_cpus=1,  # Reading and decoding images is CPU-heavy.
    memory=4 * 1024**3,  # Give Ray a hint about how much memory this task needs.
)
```

!!! note ""
    This function is most efficient when the batch of tiles belongs to a single slide and the tiles are stored sequentially. This allows it to maximize cache usage and minimize repeated I/O operations.


Next, you'll define a simple `filter_tissue` function. For this tutorial, we'll use a basic but effective heuristic: if the standard deviation of a tile's pixel values is above a certain threshold, we assume it contains tissue. Tiles that are mostly one color (like the white background) will have a very low standard deviation.

You then use `.filter()` to apply this function and keep only the rows that return `True`.

```python
tissue_tiles = tiles_with_pixels.filter(lambda row: row["tile"].std() > 8)
```


### Step 5: Saving the Results

You're almost there! The final step is to write the filtered tile information to disk. The image data itself (`row["tile"]`) can make the dataset very large, and you probably don't need to save the raw pixels. Additionally, you can omit redundant metadata (like `tile_extent`) that is already saved in the `slides` Parquet dataset.

So, you'll drop columns that are not needed for the final output.

```python
tissue_tiles = tissue_tiles.drop_columns(
    ["tile", "path", "level", "tile_extent_x", "tile_extent_y"]
)

tissue_tiles.write_parquet("tiles")
```


## Wrapping up

Congratulations! You've just built a scalable tiling pipeline. You now have:

-   `slides/`: A Parquet dataset where each row contains the metadata and unique ID for a whole-slide image.
-   `tiles/`: A Parquet dataset where each row represents a single *tissue tile*, with its parent slide's ID and its coordinates.

You've learned how to think about slide processing in a tabular way, how to use Ray Data to parallelize your work, and how to build a pipeline step-by-step.