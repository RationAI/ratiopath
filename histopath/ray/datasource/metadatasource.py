from typing import Iterator

import pyarrow
from ray.data.block import Block

from histopath.ray.datasource.abstract_metadatasource import AbstractMetaDatasource

from .ometypes_metadatasource import OmeTypesMetaDatasource
from .openslide_metadatasource import OpenSlideMetaDatasource

# Combined file extensions from both sources
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


class MetaDatasource(AbstractMetaDatasource):
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
        self.tile_extent = tile_extent
        self.stride = stride

    def _is_ome_tiff(self, path: str) -> bool:
        """Check if the file is an OME-TIFF based on file extension."""
        return path.lower().endswith((".ome.tiff", ".ome.tif"))

    def _read_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        """Read metadata using the appropriate datasource based on file type."""
        datasource: OmeTypesMetaDatasource | OpenSlideMetaDatasource
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
