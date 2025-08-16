import numpy as np


def closest_level(
    mpp: float | tuple[float, float],
    slide_mpp: float | tuple[float, float],
    downsamples: list[float],
) -> int:
    scale_factor = np.mean(np.asarray(mpp) / np.asarray(slide_mpp))

    return np.abs(np.asarray(downsamples) - scale_factor).argmin().item()
