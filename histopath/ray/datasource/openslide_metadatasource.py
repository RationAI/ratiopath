from typing import Any, Iterator

import numpy as np
from ray.data.block import Block
from ray.data.datasource import FileBasedDatasource


class OpenSlideMetaDatasource(FileBasedDatasource):
    def __init__(
        self,
        paths: str | list[str],
        *,
        mpp: float | None,
        level: int | None,
        tile_extend: Any,
        stride: Any,
    ) -> None:
        super().__init__(paths)

        assert (mpp is not None) != (level is not None), (
            "Exactly one of 'mpp' or 'level' must be provided, not both or neither."
        )

        self.desired_mpp = mpp
        self.desired_level = level
        self.tile_extent = tile_extend
        self.stride = stride

    def _read_stream(self, f, path) -> Iterator[Block]:
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

        from histopath.openslide import OpenSlide

        with OpenSlide(path) as slide:
            if self.desired_level is not None:
                level = self.desired_level
            else:
                assert self.desired_mpp is not None
                level = slide.closest_level(self.desired_mpp)
            mpp_x, mpp_y = slide.slide_resolution(level)

            extent = slide.level_dimensions[level]
            tile_extent = np.broadcast_to(self.tile_extent, 2)
            stride = np.broadcast_to(self.stride, 2)

        builder = DelegatingBlockBuilder()
        item = {
            "path": path,
            "extent_x": extent[0],
            "extent_y": extent[1],
            "tile_extent_x": tile_extent[0],
            "tile_extent_y": tile_extent[1],
            "stride_x": stride[0],
            "stride_y": stride[1],
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "level": level,
        }
        builder.add(item)
        yield builder.build()
