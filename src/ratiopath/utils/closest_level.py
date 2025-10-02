import numpy as np


def closest_level(
    mpp: float | tuple[float, float],
    slide_mpp: float | tuple[float, float],
    downsamples: list[float],
) -> int:
    """Finds the closest downsample level to the given MPP (microns per pixel) value.

    Args:
        mpp: The MPP value(s) to match.
        slide_mpp: The MPP value(s) of the slide.
        downsamples: A list of available downsample levels.

    Returns:
        The index of the closest downsample level.
    """
    scale_factor = np.mean(np.asarray(mpp) / np.asarray(slide_mpp))

    return np.abs(np.asarray(downsamples) - scale_factor).argmin().item()
