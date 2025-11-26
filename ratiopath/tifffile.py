import numpy as np
import tifffile


class TiffFile(tifffile.TiffFile):  # type: ignore [misc]
    """A wrapper around the TiffFile library to provide additional functionality."""

    def __enter__(self) -> "TiffFile":
        """Handles entering the context manager."""
        return self

    def levels(self) -> int:
        """Returns the number of levels in the OME-TIFF file.

        Returns:
            The number of levels in the OME-TIFF file.
        """
        return len(self.series[0].pages)

    def slide_resolution(self, level: int) -> tuple[float, float]:
        """Returns the physical resolution (µm/px) of the OME-TIFF file at the given level.

        Args:
            level: The level of the OME-TIFF file to get the MPP.

        Returns:
            The (x, y) physical resolution of the OME-TIFF file in µm/px.
        """
        page = self.series[0].pages[level]

        # Tag 282/283 are Rational (Num, Denom)
        x_res: tuple[int, int] = page.tags["XResolution"].value  # type: ignore [union-attr]
        y_res: tuple[int, int] = page.tags["YResolution"].value  # type: ignore [union-attr]
        # Tag 296 is ResolutionUnit
        unit: tifffile.RESUNIT = page.tags["ResolutionUnit"].value  # type: ignore [union-attr]

        # Resolve value (numerator / denominator)
        x_val = x_res[0] / x_res[1]
        y_val = y_res[0] / y_res[1]

        match unit:
            case 2:  # Inch
                return (25400 / x_val, 25400 / y_val)
            case 3:  # Centimeter
                return (10000 / x_val, 10000 / y_val)
            case 4:  # Milimeter
                return (1000 / x_val, 1000 / y_val)
            case 5:  # Micrometer
                return (x_val, y_val)
            case _:
                raise ValueError(f"Unsupported ResolutionUnit: {unit}")

    def closest_level(self, mpp: float | tuple[float, float]) -> int:
        """Finds the closest OME-TIFF level to match the desired MPP.

        This method compares the desired MPP (µm/px) with the MPP of the
        available levels in the OME-TIFF file and selects the level with the closest match.

        Args:
            mpp: The desired µm/px value.

        Returns:
            The index of the level with the closest µm/px resolution to the desired value.
        """
        available_levels_mpp = [
            self.slide_resolution(level) for level in range(self.levels())
        ]

        return (
            np.abs(np.asarray(available_levels_mpp) - np.asarray(mpp))
            .mean(axis=1)
            .argmin()
            .item()
        )
