import pyvips

from ratiopath.masks.vips_filters.typed import Res
from ratiopath.masks.vips_filters.vips_filter import VipsFilter


# The default value 10 is based on empirical observations.
DISC_FACTOR = 10


class VipsMorphology(VipsFilter):
    """Base class for morphological operations using disk structuring elements."""

    def __init__(self, disc_factor: int = DISC_FACTOR) -> None:
        self.disc_factor = disc_factor

    def _disc_size(self, mpp: Res) -> int:
        return round(self.disc_factor / (sum(mpp) / len(mpp)))

    def _disc_object(self, mpp: Res) -> pyvips.Image:
        disc_size = self._disc_size(mpp)

        return (
            pyvips.Image.black(2 * disc_size + 1, 2 * disc_size + 1) + 128
        ).draw_circle(255, disc_size, disc_size, disc_size, fill=True)


class VipsOpening(VipsMorphology):
    """Applies an opening operation to an image using a disk structuring element."""

    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        disc = self._disc_object(mpp)

        image = image.morph(disc, pyvips.enums.OperationMorphology.ERODE)
        image = image.morph(disc, pyvips.enums.OperationMorphology.DILATE)
        return image, mpp


class VipsClosing(VipsMorphology):
    """Applies a closing operation to an image using a disk structuring element."""

    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        disc = self._disc_object(mpp)

        image = image.morph(disc, pyvips.enums.OperationMorphology.DILATE)
        image = image.morph(disc, pyvips.enums.OperationMorphology.ERODE)
        return image, mpp
