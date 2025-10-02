from abc import abstractmethod

import pyvips
from numpy.typing import NDArray
from skimage.filters import threshold_multiotsu

from ratiopath.masks.vips_filters.typed import Res
from ratiopath.masks.vips_filters.vips_filter import VipsFilter


class VipsMultiOtsu(VipsFilter):
    @abstractmethod
    def apply_threshold(
        self, image: pyvips.Image, thresholds: list[float]
    ) -> pyvips.Image: ...

    @abstractmethod
    def otsu_threshold(self, hist: NDArray) -> list[float]: ...

    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        thresholds = self.otsu_threshold(image.hist_find().numpy()[0])
        return self.apply_threshold(image, thresholds), mpp


class VipsOtsu(VipsMultiOtsu):
    """Applies Otsu thresholding to an image.

    The Otsu thresholding method is used to separate the image into two classes: background and foreground.
    We use multi-Otsu to separate the image into 4 classes as there are usually up to 4 dominant colors in the image.
    Then we take the last class as the background and the rest as the foreground.
    """

    def apply_threshold(
        self, image: pyvips.Image, thresholds: list[float]
    ) -> pyvips.Image:
        return image < thresholds[2]

    def otsu_threshold(self, hist: NDArray) -> list[float]:
        return threshold_multiotsu(hist=hist, classes=4)
