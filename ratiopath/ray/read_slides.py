from typing import Any, Literal

import pyarrow.fs

import ray
from ratiopath.ray.datasource.slide_metadatasource import (
    SlideFileMetadataProvider,
    SlideMetaDatasource,
)
from ray.data import Dataset
from ray.data.datasource import BaseFileMetadataProvider, PathPartitionFilter
from ray.data.datasource.file_based_datasource import (
    FileShuffleConfig,
    _validate_shuffle_arg,
)
from ray.data.datasource.partitioning import Partitioning


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


def read_slides(
    paths: str | list[str],
    *,
    tile_extent: int | tuple[int, int],
    stride: int | tuple[int, int],
    mpp: float | None = None,
    level: int | None = None,
    filesystem: pyarrow.fs.FileSystem | None = None,
    ray_remote_args: dict[str, Any] | None = None,
    meta_provider: BaseFileMetadataProvider | None = None,
    partition_filter: PathPartitionFilter | None = None,
    partitioning: Partitioning | None = None,
    shuffle: Literal["files"] | FileShuffleConfig | None = None,
    ignore_missing_paths: bool = False,
    file_extensions: list[str] | None = FILE_EXTENSIONS,
    concurrency: int | None = None,
    override_num_blocks: int | None = None,
) -> Dataset:
    """Creates a :class:`~ray.data.Dataset` from whole slide image files.

    This function reads metadata from whole slide image (WSI) files and creates a
    Ray Dataset where each row corresponds to a single slide. The dataset contains
    metadata required for subsequent tiled processing, such as slide dimensions,
    resolution (MPP), and tiling parameters.

    It automatically selects the best slide level based on the specified `mpp`
    (microns per pixel) or uses the given `level`.

    Examples:
        Read a single slide and create a metadata dataset.

        >>> import ray
        >>> from ratiopath.ray import read_slide
        >>> ds = read_slide(  # doctest: +SKIP
        ...     "path/to/slide.svs",
        ...     tile_extent=256,
        ...     stride=256,
        ...     mpp=0.5,
        ... )
        >>> ds.schema()  # doctest: +SKIP
        Column         Type
        ------         ----
        path           string
        extent_x       int64
        extent_y       int64
        tile_extent_x  int64
        tile_extent_y  int64
        stride_x       int64
        stride_y       int64
        mpp_x          double
        mpp_y          double
        level          int64
        downsample     double

    Args:
        paths: A single file path or a list of file paths to whole slide images.
        tile_extent: The size of the tiles to be generated, as `(width, height)`.
            If a single integer is provided, it's used for both dimensions.
        stride: The step size between consecutive tiles, as `(x_stride, y_stride)`.
            If a single integer is provided, it's used for both dimensions.
        mpp: The desired microns per pixel. The datasource will select the slide
            level with the closest MPP. Exactly one of `mpp` or `level` must be
            provided.
        level: The desired slide level to use. Exactly one of `mpp` or `level`
            must be provided.
        filesystem: The PyArrow filesystem implementation to read from. If not
            provided, it will be inferred from the file paths.
        ray_remote_args: kwargs passed to :func:`ray.remote` in the read tasks.
        meta_provider: Custom metadata providers may be able to resolve file metadata
            more quickly and/or accurately. In most cases you do not need to set this
            parameter.
        partition_filter: A filter to read only selected partitions of a dataset.
        partitioning: A :class:`~ray.data.datasource.partitioning.Partitioning` object
            that describes how paths are organized.
        shuffle: If set to "files", randomly shuffles the input file order.
        ignore_missing_paths: If `True`, ignores any file paths that don't exist.
        file_extensions: A list of file extensions to filter files by. If `None`,
            it uses the default list of supported slide formats.
        concurrency: The maximum number of Ray tasks to run concurrently.
        override_num_blocks: Override the number of output blocks from all read tasks.

    Returns:
        A :class:`~ray.data.Dataset` where each row contains the metadata for one
        slide, ready for tiling operations.
    """
    _validate_shuffle_arg(shuffle)

    if meta_provider is None:
        meta_provider = SlideFileMetadataProvider()

    datasource = SlideMetaDatasource(
        paths,
        tile_extent=tile_extent,
        stride=stride,
        mpp=mpp,
        level=level,
        filesystem=filesystem,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        file_extensions=file_extensions,
    )
    return ray.data.read_datasource(
        datasource,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )
