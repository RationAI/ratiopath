from collections.abc import Callable
from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import combine_stains, separate_stains


class StainAugmentor(ImageOnlyTransform):
    """Applies stain augmentation to histopathological images.

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
        """Initializes StainAugmentor.

        Args:
            conv_matrix (Callable[[np.ndarray], np.ndarray] | np.ndarray): Stain matrix for stain separation.
                Can be a fixed matrix or a callable that returns a matrix from an image.
            alpha (float): Multiplicative factor range for stain augmentation.
            beta (float): Additive factor range for stain augmentation.
            **kwargs (Any): Keyword arguments for ImageOnlyTransform.
        """
        super().__init__(**kwargs)
        self.conv_matrix = conv_matrix
        self.alpha = alpha
        self.beta = beta

        if isinstance(self.conv_matrix, np.ndarray):
            self.inv_conv_matrix = np.linalg.inv(self.conv_matrix)

    def apply(
        self,
        img: np.ndarray,
        conv_matrix: np.ndarray,
        inv_conv_matrix: np.ndarray,
        alphas: list[float],
        betas: list[float],
        **params: dict[str, Any],
    ) -> np.ndarray:
        stains = separate_stains(img, inv_conv_matrix)

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
        inv_conv_matrix = (
            self.inv_conv_matrix
            if hasattr(self, "inv_conv_matrix")
            else np.linalg.inv(conv_matrix)
        )

        return {
            "conv_matrix": conv_matrix,
            "inv_conv_matrix": inv_conv_matrix,
            "alphas": [
                self.py_random.uniform(1 - self.alpha, 1 + self.alpha)
                for _ in range(conv_matrix.shape[-1])
            ],
            "betas": [
                self.py_random.uniform(-self.beta, self.beta)
                for _ in range(conv_matrix.shape[-1])
            ],
        }
