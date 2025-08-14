from sys import getsizeof
from typing import Iterator

import numpy as np
import pyarrow
from ray.data.block import Block
from ray.data.datasource import FileBasedDatasource

from histopath.utils.closest_level import closest_level

FILE_EXTENSIONS = [
    # OpenSlide formats
    "svs",
    "tif",
    "dcm",
    "ndpi",
    "vms",
    "vmu",
    "scn",
    "mrxs",
    "tiff",
    "svslide",
    "bif",
    "czi",
    # OME-TIFF formats
    "ome.tiff",
    "ome.tif",
]


class SlideMetaDatasource(FileBasedDatasource):
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

    def _read_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            return self._read_ome_stream(f, path)
        return self._read_openslide_stream(f, path)

    def _read_ome_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        from ome_types import from_xml
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            assert hasattr(tif, "ome_metadata") and tif.ome_metadata
            metadata = from_xml(tif.ome_metadata)

        base_px = metadata.images[0].pixels
        if base_px.physical_size_x is None or base_px.physical_size_y is None:
            raise ValueError("Physical size (MPP) is not available in the metadata.")

        if self.desired_level is not None:
            level = self.desired_level
        else:
            assert self.desired_mpp is not None
            level = closest_level(
                self.desired_mpp,
                (base_px.physical_size_x, base_px.physical_size_y),
                [base_px.size_x / img.pixels.size_x for img in metadata.images],
            )

        px = metadata.images[level].pixels
        mpp_x = px.physical_size_x
        mpp_y = px.physical_size_y
        extent = (px.size_x, px.size_y)
        downsample = metadata.images[0].pixels.size_x / px.size_x

        if mpp_x is None or mpp_y is None:
            raise ValueError("Physical size (MPP) is not available in the metadata.")

        yield self._build_block(path, extent, (mpp_x, mpp_y), level, downsample)

    def _read_openslide_stream(
        self, f: pyarrow.NativeFile, path: str
    ) -> Iterator[Block]:
        from histopath.openslide import OpenSlide

        with OpenSlide(path) as slide:
            if self.desired_level is not None:
                level = self.desired_level
            else:
                assert self.desired_mpp is not None
                level = slide.closest_level(self.desired_mpp)
            mpp_x, mpp_y = slide.slide_resolution(level)

            extent = slide.level_dimensions[level]
            downsample = slide.level_downsamples[level]

        yield self._build_block(path, extent, (mpp_x, mpp_y), level, downsample)

    def _build_block(
        self,
        path: str,
        extent: tuple[int, int],
        mpp: tuple[float, float],
        level: int,
        downsample: float,
    ) -> Block:
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

        builder = DelegatingBlockBuilder()
        item = {
            "path": path,
            "extent_x": extent[0],
            "extent_y": extent[1],
            "tile_extent_x": self.tile_extent[0],
            "tile_extent_y": self.tile_extent[1],
            "stride_x": self.stride[0],
            "stride_y": self.stride[1],
            "mpp_x": mpp[0],
            "mpp_y": mpp[1],
            "level": level,
            "downsample": downsample,
        }
        builder.add(item)
        return builder.build()

    def _rows_per_file(self) -> int:
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
            "downsample": 0.0,
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
