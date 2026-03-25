import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class JSONMultiClassMask:
    """Parses JSON annotations and renders them into a multi-class mask array.

    Expects class to value mapping on input.
    """

    def __init__(
        self,
        json_path: Path,
        mask_size: tuple[int, int],
        scale_factor: float,
        target_groups: dict[str, int],
    ):
        self.json_path = json_path
        self.mask_size = mask_size  # (width, height)
        self.scale = scale_factor
        self.target_groups = target_groups

    def __call__(self) -> np.ndarray:
        mask_img = Image.new("L", self.mask_size, 0)
        draw = ImageDraw.Draw(mask_img)

        with open(self.json_path) as f:
            data = json.load(f)

        for item in data.get("items", []):
            class_name = item.get("name")

            if class_name in self.target_groups:
                coords = item.get("coordinates", [])
                if not coords:
                    continue

                fill_value = self.target_groups[class_name]
                scaled_coords = [(c[0] * self.scale, c[1] * self.scale) for c in coords]
                draw.polygon(scaled_coords, fill=fill_value)

        return np.array(mask_img)
