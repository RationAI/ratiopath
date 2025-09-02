import pyvips

from ratiopath.masks.vips_filters.typed import Res
from ratiopath.masks.vips_filters.vips_filter import VipsFilter


class VipsGrayScaleFilter(VipsFilter):
    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        return image.colourspace(pyvips.enums.Interpretation.B_W).bandsplit()[0], mpp


class VipsSaturationFilter(VipsFilter):
    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        # Extract saturation channel
        return image.sRGB2HSV().bandsplit()[1], mpp
