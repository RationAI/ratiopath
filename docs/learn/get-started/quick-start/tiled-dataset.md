# Loading a Tiled Dataset

In the [tiling tutorial](./tiling.md), we discussed how to create a tiled dataset from massive files and save the resulting metadata to disk. Now, let's explore how to efficiently load this tiled dataset for training or inference.

Before writing the data loaders, we must consider the structure and approximate size of our dataset. A tiled dataset typically consists of two highly interrelated components:

1. **The Parent Dataset:** Contains high-level metadata about the source files (e.g., Whole Slide Image file paths, original dimensions, patient IDs, or slide-level labels).
2. **The Tile Dataset:** Contains metadata about the individual chunks derived from those parents (e.g., `x` and `y` coordinates, the parent `slide_id`, and sometimes precomputed tile embeddings).

If your Parquet files are small enough, you can safely load the entire dataset into RAM using Pandas. However, in digital pathology and large-scale computer vision, a tile dataset can easily span hundreds of gigabytes across multiple Parquet partitions. Loading this entirely into memory will crash your system.

Let's build our data loading pipeline from the ground up to handle this efficiently.

-----

## 1. The Core Building Block: `TileDataset`

At the lowest level, we need a standard PyTorch `Dataset` that represents a single Whole Slide Image. Its job is simple: take a list of tile coordinates and fetch the actual pixel data (or precomputed embeddings) for those coordinates.

In our WSI use case, we use the `openslide` library to dynamically read pixel patches from the WSIs based on the `x` and `y` metadata.

```python
from pathlib import Path

import numpy as np
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from ratiopath.openslide import OpenSlide


class TileDataset(Dataset):
    """Lazily reads pre-computed tiles from a single Whole Slide Image."""

    def __init__(
        self,
        slide_path: str | Path,
        level: int,
        extent_x: int,
        extent_y: int,
        tiles: HFDataset,
    ) -> None:
        super().__init__()
        self.slide_path = Path(slide_path)
        self.tiles = tiles  # We will discuss how to efficiently provide this next
        self.level = level
        self.extent_x = extent_x
        self.extent_y = extent_y

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> np.ndarray:
        tile = self.tiles[idx]

        # Extract coordinates
        x, y = int(tile["x"]), int(tile["y"])

        # Read the region dynamically from the slide
        with OpenSlide(self.slide_path) as slide:
            return slide.read_tile(x, y, self.extent_x, self.extent_y, self.level)

```

Notice that our `TileDataset` expects a `tiles` object containing the metadata. If we pass a standard Pandas DataFrame here, our RAM will quickly max out as we scale up to thousands of slides.

-----

## 2. Managing the Metadata: The Hugging Face `datasets` Backend

To feed our `TileDataset` without crashing our system, we use the Hugging Face [datasets](https://huggingface.co/docs/datasets/index) library. It acts as our `HFDataset` type hint above and is vastly superior to standard Pandas DataFrames for deep learning because of how it handles memory via **lazy loading**.

**How it works:**
Parquet is a heavily compressed, columnar storage format. It is great for saving disk space but terrible for the random row access required by PyTorch (`dataset[idx]`). When you load a Parquet file using Hugging Face `datasets`, the library translates the Parquet data into an uncompressed **Apache Arrow** format on your disk. It then utilizes **memory mapping** (`mmap`) to treat that file on your hard drive as if it were in your RAM.

**Why it is efficient:**

  * **Zero RAM Overhead:** You can interact with a 200GB dataset while consuming mere megabytes of actual RAM.
  * **O(1) Random Access:** Reading a specific row coordinate for our `TileDataset` is virtually instantaneous.
  * **Smart Caching:** When you filter the massive tile dataset to find only the chunks belonging to a specific slide, Hugging Face streams the data, finds the matches, and caches the view on disk.

-----

## 3. The Orchestrator: `SlideDataset`

Now we need a unified approach that links our parent metadata (the slides) with our lazily-loaded tile metadata (the patches). We achieve this through **relative tile splitting**—iterating through the parent metadata and dynamically filtering the massive Hugging Face tile dataset to extract only the chunks relative to that specific slide.

By utilizing PyTorch's `ConcatDataset`, we can seamlessly chain our individual `TileDataset` instances together into one massive, unified training set.

```python
import pyarrow.compute as pc
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import ConcatDataset


class SlideDataset(ConcatDataset[TileDataset]):
    """A unified PyTorch dataset that links parent slide metadata with tile metadata."""

    def __init__(
        self,
        slides_parquet_path: str,
        tiles_parquet_path: str,
    ) -> None:
        # 'train' is the default split name for Hugging Face datasets.
        self.slides_dataset = load_dataset("parquet", data_files=slides_parquet_path, split="train")
        # Sort by slide_id for much faster filtering
        self.tiles_dataset = load_dataset("parquet", data_files=tiles_parquet_path, split="train").sort("slide_id")

        self._slide_id_to_indices = self._build_tile_index(self.tiles_dataset)

        datasets = [
            TileDataset(
                slide_path=slide["path"],
                level=slide["level"],
                extent_x=slide["extent_x"],
                extent_y=slide["extent_y"],
                tiles=self.filter_tiles_by_slide(slide["slide_id"]),
            )
            for slide in self.slides_dataset
        ]

        super().__init__(datasets)

    
    @staticmethod
    def _build_tile_index(tiles: HFDataset) -> dict[str, range]:
        """Creates a fast lookup table for slide indices.

        This function builds a mapping from `slide_id` to the range of indices in the
        `tiles` dataset that correspond to that slide. It assumes the dataset is sorted.
        
        Args:
            tiles: A dataset containing a `slide_id` column, sorted by `slide_id`.

        Returns:
            A dictionary mapping each `slide_id` to a range of indices.
        """
        if len(tiles) == 0:
            return {}

        # Get the underlying Arrow table (zero-copy)
        table = tiles.data.table
        slide_ids = table.column("slide_id")

        # Since the dataset is sorted by 'slide_id', we can use
        # run-end encoding to find group boundaries efficiently.
        run_ends = pc.run_end_encode(slide_ids)

        values = run_ends.field("values")
        ends = run_ends.field("run_ends")

        index_map = {}
        current_offset = 0

        for sid, end in zip(values.to_pylist(), ends.to_pylist()):
            index_map[sid] = range(current_offset, end)
            current_offset = end

        return index_map

    def filter_tiles_by_slide(self, slide_id: str) -> HFDataset:
        """Returns a view of the dataset using a slice or indices.

        This uses the precomputed `_slide_id_to_indices` mapping to efficiently 
        retrieve the relevant tiles without copying data.
        """
        tile_range = self._slide_id_to_indices.get(slide_id, range(0))
        return self.tiles_dataset.select(tile_range)

```

### Using the Dataset

Once constructed, you can pass this `SlideDataset` directly into a standard PyTorch `DataLoader`. PyTorch will automatically calculate the cumulative length and map global batch indices to the correct underlying slide and tile.

```python
from torch.utils.data import DataLoader

# Build the dataset
full_dataset = SlideDataset(
    slides_parquet_path="data/slides_meta.parquet",
    tiles_parquet_path="data/tiles_meta.parquet",
)

# Load with multiprocessing for high throughput
dataloader = DataLoader(
    full_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=8
)
```