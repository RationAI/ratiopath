from collections.abc import Iterator
from sys import getsizeof
from typing import Any

import numpy as np
import pyarrow

from ratiopath.utils.closest_level import closest_level
from ray.data.block import Block, BlockMetadata
from ray.data.datasource import FileBasedDatasource
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider


class SlideMetaDatasource(FileBasedDatasource):
    """A Ray datasource that extracts metadata from whole slide image files.

    This datasource reads whole slide image files and extracts metadata required for
    tiled processing. For each slide, it determines the best level to use based on
    the provided `mpp` (microns per pixel) or `level`, and extracts properties
    like dimensions, MPP, and downsampling factor for that level.

    The output of this datasource is a Ray Dataset where each row corresponds to a
    single slide and contains the necessary metadata for subsequent tiling operations.

    Supported file formats:
        - OpenSlide: `svs`, `tif`, `dcm`, `ndpi`, `vms`, `vmu`, `scn`, `mrxs`, `tiff`, `svslide`, `bif`, `czi`
        - OME-TIFF: `ome.tiff`, `ome.tif`

    Output columns:
        - path (str): Path to the slide file.
        - extent_x (int): Width of the selected slide level in pixels.
        - extent_y (int): Height of the selected slide level in pixels.
        - tile_extent_x (int): Width of the tiles to be extracted.
        - tile_extent_y (int): Height of the tiles to be extracted.
        - stride_x (int): Horizontal stride for tiling.
        - stride_y (int): Vertical stride for tiling.
        - mpp_x (float): Microns per pixel in the x-direction for the selected level.
        - mpp_y (float): Microns per pixel in the y-direction for the selected level.
        - level (int): The selected slide level.
        - downsample (float): The downsample factor for the selected level.
    """

    def __init__(
        self,
        paths: str | list[str],
        *,
        mpp: float | None = None,
        level: int | None = None,
        tile_extent: int | tuple[int, int],
        stride: int | tuple[int, int],
        **file_based_datasource_kwargs: Any,
    ) -> None:
        """Initializes the SlideMetaDatasource.

        Args:
            paths: A path or list of paths to whole slide image files.
            mpp: The desired microns per pixel. The datasource will select the slide
                level with the closest MPP. Exactly one of `mpp` or `level` must be
                provided.
            level: The desired slide level to use. Exactly one of `mpp` or `level`
                must be provided.
            tile_extent: The size of the tiles to be generated, as (width, height).
                If a single integer is provided, it's used for both dimensions.
            stride: The step size between consecutive tiles, as (x_stride, y_stride).
                If a single integer is provided, it's used for both dimensions.
            **file_based_datasource_kwargs: Additional keyword arguments passed to
                the base `FileBasedDatasource`.
        """
        super().__init__(paths, **file_based_datasource_kwargs)

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
        from ratiopath.openslide import OpenSlide

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


class SlideFileMetadataProvider(DefaultFileMetadataProvider):
    def _get_block_metadata(  # type: ignore[override]
        self,
        paths: list[str],
        *,
        rows_per_file: int | None,
        file_sizes: list[int | None],
    ) -> BlockMetadata:
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
        base_row_size = getsizeof(sample_item)
        for k, v in sample_item.items():
            base_row_size += getsizeof(k)
            base_row_size += getsizeof(v)

        # Estimate size for each path and create a new list of file sizes.
        estimated_file_sizes = [base_row_size + getsizeof(p) for p in paths]

        return super()._get_block_metadata(
            paths,
            rows_per_file=rows_per_file,
            file_sizes=estimated_file_sizes,  # type: ignore[arg-type]
        )
