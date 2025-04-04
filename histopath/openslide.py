import numpy as np
import openslide
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y


class OpenSlide(openslide.OpenSlide):
    def closest_level(self, mpp: float | tuple[float, float]) -> int:
        """Finds the closest slide level to match the desired MPP.

        This method compares the desired MPP (µm/px) with the MPP of the
        available levels in the slide and selects the level with the closest match.

        Args:
            mpp: The desired µm/px value.

        Returns:
            The index of the level with the closest µm/px resolution to the desired value.
        """
        slide_mpp_x = float(self.properties[PROPERTY_NAME_MPP_X])
        slide_mpp_y = float(self.properties[PROPERTY_NAME_MPP_Y])

        scale_factor = np.mean(np.asarray(mpp) / np.asarray([slide_mpp_x, slide_mpp_y]))

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

        return tuple(
            self.level_downsamples[level] * np.asarray((slide_mpp_x, slide_mpp_y))
        )

    def adjust_read_coords(
        self, coords: tuple[int, int], level: int
    ) -> tuple[int, int]:
        """Adjusts the coordinates to read the region at the specified level.

        Args:
            coords: The coordinates to adjust.
            level: The level of the slide to adjust the coordinates for.

        Returns:
            The adjusted coordinates.
        """
        return tuple(
            np.round(np.asarray(coords) * self.level_downsamples[level])
        )
