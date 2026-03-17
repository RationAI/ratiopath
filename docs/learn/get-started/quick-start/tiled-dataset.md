# Loading a Tiled Dataset

In the [tiling tutorial](./tiling.md), we discussed how to create a tiled dataset from massive files and save the resulting metadata to disk. Now, let's explore how to efficiently load this tiled dataset for training or inference.

Before writing the data loaders, we must consider the structure and approximate size of our dataset. A tiled dataset typically consists of two highly interrelated components:

1. **The Parent Dataset:** Contains high-level metadata about the source files (e.g., Whole Slide Image file paths, original dimensions, patient IDs, or slide-level labels).
2. **The Tile Dataset:** Contains metadata about the individual chunks derived from those parents (e.g., `x` and `y` coordinates, the parent `slide_id`, and sometimes precomputed tile embeddings).

If your Parquet files are small enough, you can safely load the entire dataset into RAM using Pandas. However, in digital pathology and large-scale computer vision, a tile dataset can easily span hundreds of gigabytes across multiple Parquet partitions. Loading this entirely into memory will crash your system. Instead, we rely on **lazy loading** techniques to fetch only the necessary data points exactly when the model needs them.

---

## 1. Lazy Loading: The Hugging Face `datasets` Backend

To handle massive tabular metadata, we use the Hugging Face [datasets](https://huggingface.co/docs/datasets/index) library. It is vastly superior to standard Pandas DataFrames for deep learning data loaders because of how it handles memory.

**How it works:**
Parquet is a heavily compressed, columnar storage format. It is great for saving disk space but terrible for the random row access required by PyTorch (e.g., `dataset[idx]`).

When you load a Parquet file using Hugging Face `datasets`, the library translates the Parquet data into an uncompressed **Apache Arrow** format on your disk. It then utilizes **memory mapping** (`mmap`) to treat that file on your hard drive as if it were in your RAM.

**Why it is efficient:**

* **Zero RAM Overhead:** You can interact with a 200GB dataset while consuming mere megabytes of actual RAM.
* *x`O(1)` Random Access:** Reading a specific row is virtually instantaneous.
* **Smart Caching:** When you filter the dataset to find tiles belonging to a specific slide, Hugging Face streams the data, finds the matches, and caches the result on disk.

---

## 2. The Tile Dataset (Reading Individual Tiles)

At the lowest level, we need a standard PyTorch `Dataset` that takes a subset of our tiled data eg. fetches the actual pixel data, or the precomputed embeddings.

In our WSI use case, we use the `openslide` library to dynamically read pixel patches from the WSIs based on the `x` and `y` coordinates stored in our Arrow-mapped tile dataset.

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
        self.tiles = tiles
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

---

## 3. The Main Torch Dataset (Linking Slides and Tiles)

Now we need a unified approach that combines our parent dataset (the slides) with our tile dataset (the patches). We achieve this through **relative tile splitting**—iterating through the parent metadata and dynamically filtering the massive tile dataset to extract only the chunks relative to that specific parent.

By utilizing PyTorch's `ConcatDataset`, we can seamlessly chain these individual `SlideTileDataset` instances together into one massive, unified training set.

```python
from datasets import load_dataset
from torch.utils.data import ConcatDataset


class SlideDataset(ConcatDataset[TileDataset]):
    """A unified PyTorch dataset that links parent slide metadata with tile metadata."""

    def __init__(
        self,
        slides_parquet_path: str,
        tiles_parquet_path: str,
    ) -> None:
        self.slides_dataset = load_dataset("parquet", data_files=slides_parquet_path, split="train") # Train is default split name for Hugging Face datasets, even if we don't have multiple splits
        self.tiles_dataset = load_dataset("parquet", data_files=tiles_parquet_path, split="train").sort("slide_id") # Sort by slide_id for faster filtering

        self._slide_id_to_indices = self._build_tile_index(self.tiles_dataset)

        datasets = [
            TileDataset(
                slide_path=slide["path"],
                level=slide["level"],
                extent_x=slide["extent_x"],
                extent_y=slide["extent_y"],
                tiles=self.filter_tiles_by_slide(slide.id),
            )
            for slide in self.slides_dataset
        ]

        super().__init__(datasets)

    
    @staticmethod
    def _build_tile_index(tiles: HFDataset) -> dict[str, range]:
        """Creates a fast lookup table for slide indices.

        This function builds a mapping from `slide_id` to the range of indices in the
        `tiles` dataset that correspond to that slide. It assumes that the `tiles` dataset
        is sorted by `slide_id`, which allows for efficient retrieval of tile indices
        for each slide without needing to scan the entire dataset for each slide.

        Args:
            tiles: A dataset containing a `slide_id` column, sorted by `slide_id`.

        Returns:
            A dictionary mapping each `slide_id` to a range of indices in the `tiles` dataset.
        """
        if len(tiles) == 0:
            return {}

        # Get the underlying Arrow table (zero-copy)
        table = tiles.data.table

        # Since it's sorted, we only care about where 'slide_id' changes.
        slide_ids = table.column("slide_id")

        # Find unique values and their counts
        counts = pc.value_counts(slide_ids)  # pyright: ignore[reportAttributeAccessIssue]

        index_map = {}
        current_offset = 0

        # counts is a StructArray of {values: T, counts: int64}
        for count in counts:
            pair = count.as_py()
            sid = pair["values"]
            offset = pair["counts"]

            index_map[sid] = range(current_offset, current_offset + offset)
            current_offset += offset

        return index_map

    def filter_tiles_by_slide(self, slide_id: str) -> HFDataset:
        """Returns a view of the dataset using a slice or indices.

        This function creates a view of the `self.tiles` dataset that contains only
        the tiles belonging to the specified slide. It uses the precomputed
        `_slide_id_to_indices` mapping to efficiently retrieve the relevant tiles
        without copying data.

        Args:
            slide_id: The ID of the slide to filter tiles.

        Returns:
            A view of the tiles dataset containing only the tiles for the specified slide.
        """
        tile_range = self._slide_id_to_indices.get(slide_id, range(0))
        return self.tiles_dataset.select(tile_range)

```

### Using the Dataset

Once constructed, you can pass this `ConcatDataset` directly into a standard PyTorch `DataLoader`. PyTorch will automatically calculate the cumulative length and map global batch indices to the correct underlying slide and tile.

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