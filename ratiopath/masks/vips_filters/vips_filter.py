from abc import ABC, abstractmethod
from collections.abc import Iterable

import pyvips

from ratiopath.masks.vips_filters.typed import Res


class VipsFilter(ABC):
    @abstractmethod
    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]: ...


class VipsCompose(VipsFilter):
    def __init__(self, filters: Iterable[VipsFilter]) -> None:
        self.filters = filters

    def __call__(self, image: pyvips.Image, mpp: Res) -> tuple[pyvips.Image, Res]:
        for filter in self.filters:
            image, mpp = filter(image, mpp)
        return image, mpp
