import os
from sys import getsizeof
from typing import Iterator

import numpy as np
import pyarrow
from ray.data.block import Block
from ray.data.datasource import FileBasedDatasource

from .ometypes_metadatasource import OmeTypesMetaDatasource
from .openslide_metadatasource import OpenSlideMetaDatasource

# Combined file extensions from both sources
FILE_EXTENSIONS = [
    # OpenSlide formats
    "svs", "tif", "dcm", "ndpi", "vms", "vmu", "scn", "mrxs", 
    "tiff", "svslide", "bif", "czi",
    # OME-TIFF formats
    "ome.tiff", "ome.tif",
]


class MetaDatasource(FileBasedDatasource):
    """Unified datasource for reading metadata from both OpenSlide and OME-TIFF files.

    This datasource automatically chooses between OpenSlideMetaDatasource and 
    OmeTypesMetaDatasource based on the file extension. It provides a consistent
    interface for reading metadata from various microscopy image formats.

    Args:
        paths: Path(s) to the image files.
        mpp: Desired resolution in microns per pixel. If provided, `level` must be None.
        level: Desired level of the slide. If provided, `mpp` must be None.
        tile_extent: Size of the tiles to be generated from the slide.
        stride: Stride for tiling the slide.
        **file_based_datasource_kwargs: Additional arguments for the FileBasedDatasource.

    Raises:
        AssertionError: If both `mpp` and `level` are provided or if neither is provided.

    Example:
        >>> from histopath.ray.datasource.metadatasource import MetaDatasource
        >>> datasource = MetaDatasource(
        ...     paths=["slide1.svs", "slide2.ome.tiff"],
        ...     mpp=0.25,
        ...     tile_extent=(512, 512),
        ...     stride=(256, 256),
        ... )
    """

    def __init__(
        self,
        paths: str | list[str],
        *,
        mpp: float | None = None,
        level: int | None = None,
        tile_extent: int | tuple[int, int],
        stride: int | tuple[int, int],
        **file_based_datasource_kwargs,
    ) -> None:
        super().__init__(
            paths, file_extensions=FILE_EXTENSIONS, **file_based_datasource_kwargs
        )

        assert (mpp is not None) != (level is not None), (
            "Exactly one of 'mpp' or 'level' must be provided, not both or neither."
        )

        self.desired_mpp = mpp
        self.desired_level = level
        self.tile_extent = np.broadcast_to(tile_extent, 2)
        self.stride = np.broadcast_to(stride, 2)

    def _is_ome_tiff(self, path: str) -> bool:
        """Check if the file is an OME-TIFF based on file extension."""
        return path.lower().endswith(('.ome.tiff', '.ome.tif'))

    def _read_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        """Read metadata using the appropriate datasource based on file type."""
        if self._is_ome_tiff(path):
            # Use OME-TIFF datasource
            datasource = OmeTypesMetaDatasource(
                paths=[path],
                mpp=self.desired_mpp,
                level=self.desired_level,
                tile_extent=self.tile_extent,
                stride=self.stride,
            )
        else:
            # Use OpenSlide datasource
            datasource = OpenSlideMetaDatasource(
                paths=[path],
                mpp=self.desired_mpp,
                level=self.desired_level,
                tile_extent=self.tile_extent,
                stride=self.stride,
            )
        
        # Delegate to the appropriate datasource
        yield from datasource._read_stream(f, path)

    def _rows_per_file(self) -> int:  # type: ignore[override]
        return 1

    def estimate_inmemory_data_size(self) -> int | None:
        paths = self._paths()
        if not paths:
            return 0

        # Create a sample item to calculate the base size of a single row.
        sample_item = {
            "path": "",
            "extent_x": 0,
            "extent_y": 0,
            "tile_extent_x": 0,
            "tile_extent_y": 0,
            "stride_x": 0,
            "stride_y": 0,
            "mpp_x": 0.0,
            "mpp_y": 0.0,
            "level": 0,
        }

        # Calculate the size of the dictionary structure, keys, and fixed-size values.
        base_row_size = getsizeof(sample_item)
        for k, v in sample_item.items():
            base_row_size += getsizeof(k)
            base_row_size += getsizeof(v)

        # Calculate the total size of all path strings.
        total_path_size = sum(getsizeof(p) for p in paths)

        # The total estimated size is the base size for each row plus the total size of paths.
        return base_row_size * len(paths) + total_path_size