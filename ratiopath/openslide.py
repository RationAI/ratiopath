import numpy as np
import openslide
import pyarrow as pa
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from PIL.Image import Image


class OpenSlide(openslide.OpenSlide):
    """A wrapper around the OpenSlide library to provide additional functionality."""

    def closest_level(self, mpp: float | tuple[float, float]) -> int:
        """Finds the closest slide level to match the desired MPP.

        This method compares the desired MPP (µm/px) with the MPP of the
        available levels in the slide and selects the level with the closest match.

        Args:
            mpp: The desired µm/px value.

        Returns:
            The index of the level with the closest µm/px resolution to the desired value.
        """
        scale_factor = np.mean(
            np.asarray(mpp)
            / np.array(
                [
                    float(self.properties[PROPERTY_NAME_MPP_X]),
                    float(self.properties[PROPERTY_NAME_MPP_Y]),
                ]
            )
        )

        return np.abs(np.asarray(self.level_downsamples) - scale_factor).argmin().item()

    def slide_resolution(self, level: int) -> tuple[float, float]:
        """Returns the resolution of the slide in µm/px at the given level.

        Args:
            level: The level of the slide to calculate the resolution.

        Returns:
            The [x, y] resolution of the slide in µm/px.
        """
        slide_mpp_x = float(self.properties[PROPERTY_NAME_MPP_X])
        slide_mpp_y = float(self.properties[PROPERTY_NAME_MPP_Y])

        return (
            slide_mpp_x * self.level_downsamples[level],
            slide_mpp_y * self.level_downsamples[level],
        )

    def read_region_relative(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image:
        """Reads a region from the slide with coordinates relative to the specified level.

        This method adjusts the coordinates based on the level's downsampling factor
        before reading the region from the slide.

        Args:
            location: The (x, y) coordinates at the specified level.
            level: The level of the slide to read from.
            size: The (width, height) of the region to read.

        Returns:
            The image of the requested region.
        """
        downsample = self.level_downsamples[level]
        location = (int(location[0] * downsample), int(location[1] * downsample))

        return super().read_region(location, level, size)

    def read_tile(
        self,
        x: int | pa.Scalar,
        y: int | pa.Scalar,
        extent_x: int | pa.Scalar,
        extent_y: int | pa.Scalar,
        level: int | pa.Scalar,
        background: tuple[int, int, int] | int = 255,
    ) -> np.ndarray:
        """Reads a tile from the slide at the specified coordinates and level.

        This method reads a tile from the slide based on the provided x and y
        coordinates, tile extent, and level. It also composites the tile onto a
        white background to remove any alpha channel.

        Args:
            x: The x-coordinate of the tile at level 0.
            y: The y-coordinate of the tile at level 0.
            extent_x: The width of the tile in pixels.
            extent_y: The height of the tile in pixels.
            level: The level of the slide to read from.
            background: The RGB value (0-255) to use for transparent areas. Defaults to 255 (white).

        Returns:
            The RGB image of the requested tile.

        """
        # Convert PyArrow scalars to native Python ints if needed
        x_val = x.as_py() if isinstance(x, pa.Scalar) else x
        y_val = y.as_py() if isinstance(y, pa.Scalar) else y
        extent_x_val = extent_x.as_py() if isinstance(extent_x, pa.Scalar) else extent_x
        extent_y_val = extent_y.as_py() if isinstance(extent_y, pa.Scalar) else extent_y
        level_val = level.as_py() if isinstance(level, pa.Scalar) else level

        background_broadcasted = tuple(np.broadcast_to(background, (3,)))

        rgba_region = self.read_region_relative(
            (x_val, y_val), level_val, (extent_x_val, extent_y_val)
        )

        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, background_broadcasted),  # pyright: ignore[reportAttributeAccessIssue]
            rgba_region,
        ).convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]

        return np.asarray(rgb_region)
