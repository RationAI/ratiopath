from typing import Any

import numpy as np
from openslide import OpenSlide
from PIL import Image


def openslide_tile_reader(row: dict[str, Any]) -> Any:
    with OpenSlide(row["path"]) as slide:
        rgba_region = slide.read_region(
            (row["tile_x"], row["tile_y"]),
            row["level"],
            (row["tile_extent_x"], row["tile_extent_y"]),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        row["tile"] = np.asarray(rgb_region)

    return row
