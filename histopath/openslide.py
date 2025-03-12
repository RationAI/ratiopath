import numpy as np
import openslide
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y


class OpenSlide(openslide.OpenSlide):
    def closest_level(self, mpp: float) -> int:
        """Finds the closest slide level to match the desired MPP.

        This method compares the desired MPP (µm/px) with the MPP of the
        available levels in the slide and selects the level with the closest match.

        Args:
            mpp: The desired µm/px value.

        Returns:
            The index of the level with the closest µm/px resolution to the desired value.
        """
        slide_mpp = np.array(
            [
                float(self.properties[PROPERTY_NAME_MPP_X]),
                float(self.properties[PROPERTY_NAME_MPP_Y]),
            ]
        )

        scale_factor = np.mean(mpp / slide_mpp)

        return np.abs(np.asarray(self.level_downsamples) - scale_factor).argmin().item()

    def slide_resolution(self, level: int) -> tuple[float, float]:
        """Returns the resolution of the slide in µm/px at the given level.

        Args:
            level: The level of the slide to calculate the resolution.

        Returns:
            The [x, y] resolution of the slide in µm/px.
        """
        return tuple(
            self.level_downsamples[level]
            * np.asarray(
                (
                    float(self.properties[PROPERTY_NAME_MPP_X]),
                    float(self.properties[PROPERTY_NAME_MPP_Y]),
                )
            )
        )
