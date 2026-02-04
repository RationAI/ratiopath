from pathlib import Path

import pyvips


def write_big_tiff(
    image: pyvips.Image,
    path: Path,
    mpp_x: float,
    mpp_y: float,
    tile_width: int = 512,
    tile_height: int = 512,
) -> None:
    """Saves the image as a BigTIFF file using pyvips with predefined settings.

    This function exports an image in BigTIFF format, which is ideal for applications
    that require multi-resolution pyramid TIFF files, such as xOpat. It leverages
    [`pyvips.Image.tiffsave`](https://libvips.github.io/pyvips/vimage.html#pyvips.Image.tiffsave)
    to apply specific parameters for efficient storage and optimal image tiling.

    The image is saved with the following fixed settings:
        - BigTIFF format enabled (`bigtiff=True`).
        - DEFLATE compression applied for efficient storage.
        - Image is tiled with a tile size of tile_width x tile_height.
        - Multi-level pyramid structure is used (`pyramid=True`), making the image suitable
            for fast zooming in viewers.

    Args:
        image: A pyvips Image object representing the input image.
        path: The file path where the BigTIFF mask will be saved.
        mpp_x: The horizontal resolution of the image in µm/pixel.
        mpp_y: The vertical resolution of the image in µm/pixel.
        tile_width: The width of each tile in pixels. Default is 512.
            Must match with other masks used in xOpat.
        tile_height: The height of each tile in pixels. Default is 512.
            Must match with other masks used in xOpat.
    """
    xres = 1000 / mpp_x
    yres = 1000 / mpp_y

    image.tiffsave(
        path,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=tile_width,
        tile_height=tile_height,
        xres=xres,
        yres=yres,
        pyramid=True,
    )
