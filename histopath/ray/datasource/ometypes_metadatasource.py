from sys import getsizeof
from typing import Iterator

import numpy as np
import pyarrow
import tifffile
from ray.data.block import Block
from ray.data.datasource import FileBasedDatasource

FILE_EXTENSIONS = [
    "ome.tiff",
    "ome.tif",
]


class OmeTypesMetaDatasource(FileBasedDatasource):
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
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

        with tifffile.TiffFile(path) as tif:
            # Get OME metadata
            ome_metadata = tif.ome_metadata
            
            if self.desired_level is not None:
                level = self.desired_level
            else:
                assert self.desired_mpp is not None
                level = self._closest_level(tif, self.desired_mpp)
            
            # Get the series at the specified level
            series = tif.series[0]  # Main image series
            page = series.pages[level] if level < len(series.pages) else series.pages[0]
            
            # Get physical pixel sizes from OME metadata
            mpp_x, mpp_y = self._get_physical_pixel_size(tif, level)
            
            # Get dimensions
            extent = (page.imagewidth, page.imagelength)

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
        }
        builder.add(item)
        yield builder.build()

    def _closest_level(self, tif: tifffile.TiffFile, desired_mpp: float) -> int:
        """Find the closest level to the desired MPP."""
        series = tif.series[0]
        
        # Get the base resolution
        base_mpp_x, base_mpp_y = self._get_physical_pixel_size(tif, 0)
        base_mpp = np.mean([base_mpp_x, base_mpp_y])
        
        if base_mpp == 0:
            # If no physical pixel size info, just return level 0
            return 0
        
        # Calculate scale factor needed
        scale_factor = desired_mpp / base_mpp
        
        # Find the level with the closest scale factor
        best_level = 0
        best_diff = float('inf')
        
        for level in range(len(series.pages)):
            page = series.pages[level]
            # Estimate level scale based on size ratio to level 0
            level_scale = series.pages[0].imagewidth / page.imagewidth
            diff = abs(level_scale - scale_factor)
            
            if diff < best_diff:
                best_diff = diff
                best_level = level
        
        return best_level

    def _get_physical_pixel_size(self, tif: tifffile.TiffFile, level: int) -> tuple[float, float]:
        """Get physical pixel size in micrometers."""
        try:
            # Try to get from OME metadata first
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                
                # Look for PhysicalSizeX and PhysicalSizeY in OME metadata
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels = root.find('.//ome:Pixels', ns)
                if pixels is not None:
                    size_x = pixels.get('PhysicalSizeX')
                    size_y = pixels.get('PhysicalSizeY')
                    unit_x = pixels.get('PhysicalSizeXUnit', 'µm')
                    unit_y = pixels.get('PhysicalSizeYUnit', 'µm')
                    
                    if size_x and size_y:
                        mpp_x = float(size_x)
                        mpp_y = float(size_y)
                        
                        # Convert to micrometers if needed
                        if unit_x == 'mm':
                            mpp_x *= 1000
                        elif unit_x == 'm':
                            mpp_x *= 1000000
                        
                        if unit_y == 'mm':
                            mpp_y *= 1000
                        elif unit_y == 'm':
                            mpp_y *= 1000000
                        
                        # Adjust for level downsampling
                        if level > 0:
                            series = tif.series[0]
                            downsample = series.pages[0].imagewidth / series.pages[level].imagewidth
                            mpp_x *= downsample
                            mpp_y *= downsample
                        
                        return mpp_x, mpp_y
            
            # Fallback to TIFF tags
            page = tif.series[0].pages[level]
            resolution = getattr(page, 'tags', {}).get('XResolution', None)
            if resolution and resolution.value:
                # Convert from pixels/inch to micrometers/pixel
                pixels_per_inch = resolution.value[0] / resolution.value[1]
                mpp = 25400.0 / pixels_per_inch  # 25400 micrometers per inch
                return mpp, mpp
                
        except Exception:
            pass
        
        # Default fallback
        return 0.25, 0.25  # Default 0.25 µm/pixel

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