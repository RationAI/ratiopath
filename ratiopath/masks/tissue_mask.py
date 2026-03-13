from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pyvips

    from ratiopath.masks.vips_filters import VipsFilter
    from ratiopath.masks.vips_filters.typed import Res


def tissue_mask(
    slide: pyvips.Image, mpp: Res, filter: VipsFilter | None = None
) -> tuple[pyvips.Image, Res]:
    """Generates a tissue mask from a whole-slide image (WSI) using saturation channel extraction and morphological operations, and saves the mask as a TIFF image.

    The function extracts the saturation channel from the WSI, applies Otsu thresholding to
    identify tissue regions, and performs morphological operations (closing and opening)
    to refine the mask.

    Args:
        slide: whole-slide image (WSI) pyvips file handler.
        mpp: Resolution of the mask in µm/px.
        filter: A VipsFilter object that defines the operations to be applied to the image.
            The default filter includes saturation channel extraction, Otsu thresholding, closing, and opening operations.
            The disk factor for morphological operations using the default filter is 10.

    Returns:
        The generated tissue mask as pyvips.Image.
    """
    from ratiopath.masks.vips_filters import (
        VipsClosing,
        VipsCompose,
        VipsGrayScaleFilter,
        VipsOpening,
        VipsOtsu,
    )

    if filter is None:
        filter = VipsCompose(
            [
                VipsGrayScaleFilter(),
                VipsOtsu(),
                VipsOpening(),
                VipsClosing(),
            ]
        )

    return filter(slide, mpp)
