from typing import Any, Callable

import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import combine_stains, separate_stains


class StainAugmentor(ImageOnlyTransform):
    """StainAugmentor

    Reference:
        Tellez, D., Balkenhol, M., Karssemeijer, N., Litjens, G.,
        van der Laak, J., & Ciompi, F. (2018, March). H&E stain augmentation improves
        generalization of convolutional networks for histopathological mitosis detection.
        In Medical Imaging 2018: Digital Pathology (Vol. 10581, pp. 264-270). SPIE.
        https://geertlitjens.nl/publication/tell-18-a/tell-18-a.pdf

    """

    def __init__(
        self,
        conv_matrix: Callable[[np.ndarray], np.ndarray] | np.ndarray,
        alpha: float = 0.02,
        beta: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.conv_matrix = conv_matrix
        self.alpha = alpha
        self.beta = beta

    def apply(
        self,
        img: np.ndarray,
        conv_matrix: np.ndarray,
        alphas: list[float],
        betas: list[float],
        **params: dict[str, Any],
    ) -> np.ndarray:
        stains = separate_stains(img, np.linalg.inv(conv_matrix))

        for i in range(stains.shape[-1]):
            stains[..., i] *= alphas[i]
            stains[..., i] += betas[i]

        return np.astype(combine_stains(stains, conv_matrix) * 255, np.uint8)

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        conv_matrix = (
            self.conv_matrix(data["image"])
            if callable(self.conv_matrix)
            else self.conv_matrix
        )

        return {
            "conv_matrix": conv_matrix,
            "alphas": [
                self.py_random.uniform(1 - self.alpha, 1 + self.alpha)
                for _ in range(conv_matrix.shape[-1])
            ],
            "betas": [
                self.py_random.uniform(-self.beta, self.beta)
                for _ in range(conv_matrix.shape[-1])
            ],
        }
