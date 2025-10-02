import openslide
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from PIL.Image import Image

from ratiopath.utils.closest_level import closest_level


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
        return closest_level(
            mpp,
            (
                float(self.properties[PROPERTY_NAME_MPP_X]),
                float(self.properties[PROPERTY_NAME_MPP_Y]),
            ),
            list(self.level_downsamples),
        )

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
