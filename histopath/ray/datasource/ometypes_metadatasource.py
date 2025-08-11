from typing import Iterator

import numpy as np
import pyarrow
from ome_types import OME
from ray.data.block import Block

from histopath.ray.datasource.abstract_metadatasource import AbstractMetaDatasource

FILE_EXTENSIONS = [
    "ome.tiff",
    "ome.tif",
]


class OmeTypesMetaDatasource(AbstractMetaDatasource):
    """Datasource for reading OME-TIFF metadata.

    This datasource reads metadata from OME-TIFF files and returns a block containing
    the metadata for each file. The metadata includes the slide dimensions, tile extent,
    stride, and resolution (microns per pixel) for the specified level or resolution.

    Args:
        paths: Path(s) to the OME-TIFF files.
        mpp: Desired resolution in microns per pixel. If provided, `level` must be None.
        level: Desired level of the slide. If provided, `mpp` must be None.
        tile_extent: Size of the tiles to be generated from the slide.
        stride: Stride for tiling the slide.
        **file_based_datasource_kwargs: Additional arguments for the FileBasedDatasource.

    Raises:
        AssertionError: If both `mpp` and `level` are provided or if neither is provided.

    Example:
        >>> from histopath.ray.datasource.ometypes_metadatasource import OmeTypesMetaDatasource
        >>> datasource = OmeTypesMetaDatasource(
        ...     paths=["slide1.ome.tiff", "slide2.ome.tiff"],
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

    def _read_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        from ome_types import from_xml
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            assert hasattr(tif, "ome_metadata") and tif.ome_metadata
            metadata = from_xml(tif.ome_metadata)

            if self.desired_level is not None:
                level = self.desired_level
            else:
                assert self.desired_mpp is not None
                level = self._closest_level(self.desired_mpp, metadata)

            px = metadata.images[level].pixels
            mpp_x = px.physical_size_x
            mpp_y = px.physical_size_y
            extent = (px.size_x, px.size_y)
            downsample = metadata.images[0].pixels.size_x / px.size_x

        builder = DelegatingBlockBuilder()
        item = {
            "path": path,
            "extent_x": extent[0],
            "extent_y": extent[1],
            "tile_extent_x": self.tile_extent[0],
            "tile_extent_y": self.tile_extent[1],
            "stride_x": self.stride[0],
            "stride_y": self.stride[1],
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "level": level,
            "downsample": downsample,
        }
        builder.add(item)
        yield builder.build()

    def _closest_level(self, mpp: float | tuple[float, float], metadata: OME) -> int:
        slide_mpp_x = metadata.images[0].pixels.physical_size_x
        slide_mpp_y = metadata.images[0].pixels.physical_size_y

        scale_factor = np.mean(np.asarray(mpp) / np.asarray([slide_mpp_x, slide_mpp_y]))

        level_downsamples = [
            metadata.images[0].pixels.size_x / img.pixels.size_x
            for img in metadata.images
        ]

        return np.abs(np.asarray(level_downsamples) - scale_factor).argmin().item()
