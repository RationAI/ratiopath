import numpy as np


HEMATOXYLIN = np.array([0.65, 0.70, 0.29])  # From Ruifrok & Johnston's original paper
# EOSIN = np.array([0.07, 0.99, 0.11]) # From Ruifrok & Johnston's original paper
EOSIN = np.array([0.2159, 0.8012, 0.5581])  # From http://amida13.isi.uu.nl/?q=node/69
DAB = np.array([0.27, 0.57, 0.78])  # From Ruifrok & Johnston's original paper


def make_residual_stain(stain1: np.ndarray, stain2: np.ndarray) -> np.ndarray:
    """Create a residual stain vector from two stain vectors.

    Parameters:
        stain1: A numpy array representing the first stain vector.
        stain2: A numpy array representing the second stain vector.

    Returns:
        A numpy array representing the residual stain vector.
    """
    res = np.linalg.cross(stain1, stain2)
    return res / np.linalg.norm(res)


HE = np.array(
    [HEMATOXYLIN, EOSIN, make_residual_stain(HEMATOXYLIN, EOSIN)], dtype=np.float32
)
HDAB = np.array(
    [HEMATOXYLIN, DAB, make_residual_stain(HEMATOXYLIN, DAB)], dtype=np.float32
)


def discard_pixels(
    od: np.ndarray,
    min_stain: float,
    max_stain: float,
    gray_threshold: float = np.cos(0.15),
) -> np.ndarray:
    """Discard pixels based on optical density thresholds.

    Parameters:
        od: A numpy array of optical densities for red, green, and blue channels.
        min_stain: Minimum optical density threshold.
        max_stain: Maximum optical density threshold.
        gray_threshold: Threshold for excluding very gray pixels (default is cos(0.15)).

    Returns:
        A numpy array containing the filtered optical densities for red, green, and blue channels.
    """
    keep_count = 0
    max_stain_squared = max_stain * max_stain
    sqrt3 = 1 / np.sqrt(3)

    for i in range(len(od)):
        r, g, b = od[i]
        mag_squared = r * r + g * g + b * b
        if (
            mag_squared > max_stain_squared
            or r < min_stain
            or g < min_stain
            or b < min_stain
            or mag_squared <= 0
        ):
            continue

        # Exclude very gray pixels
        if (r * sqrt3 + g * sqrt3 + b * sqrt3) / np.sqrt(mag_squared) >= gray_threshold:
            continue

        od[keep_count] = np.array([r, g, b])
        keep_count += 1

    return od[:keep_count]


def estimate_stain_vectors(
    image: np.ndarray,
    default_stain_vectors: np.ndarray,
    i0: int = 256,
    min_stain: float = 0.05,
    max_stain: float = 1.0,
    alpha: float = 0.01,
) -> np.ndarray:
    """Estimate stain vectors from an image using optical density transformation.

    Parameters:
        image: A numpy array representing the input image.
        default_stain_vectors: A numpy array of default unit stain vectors.
        i0: The intensity value for normalization.
        min_stain: Minimum optical density threshold for discarding pixels.
        max_stain: Maximum optical density threshold for discarding pixels.
        alpha: The percentage of pixels to use for estimating the stain vectors
            (default is 0.01, which corresponds to 1%).

    Returns:
        A numpy array of estimated stain vectors.

    References:
        Paper: A method for normalizing histology slides for quantitative analysis,
            M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun,
            C Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
    """
    od = -np.log10(np.maximum(image.reshape(-1, 3), 1) / i0)

    od_hat = discard_pixels(od, min_stain, max_stain)

    cov = np.cov(od_hat.T)

    _, eigvecs = np.linalg.eigh(cov)

    t_hat = od_hat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    direction_min = np.array([np.cos(min_phi), np.sin(min_phi)], dtype=np.float32)
    stain1 = eigvecs[:, 1:3].dot(direction_min).reshape(-1)
    stain1 /= np.linalg.norm(stain1)

    direction_max = np.array([np.cos(max_phi), np.sin(max_phi)], dtype=np.float32)
    stain2 = eigvecs[:, 1:3].dot(direction_max).reshape(-1)
    stain2 /= np.linalg.norm(stain2)

    cos_angle11 = np.dot(stain1, default_stain_vectors[0])
    cos_angle12 = np.dot(stain1, default_stain_vectors[1])
    cos_angle21 = np.dot(stain2, default_stain_vectors[0])
    cos_angle22 = np.dot(stain2, default_stain_vectors[1])

    if max(cos_angle12, cos_angle21) > max(cos_angle11, cos_angle22):
        stain1, stain2 = stain2, stain1

    return np.array([stain1, stain2, make_residual_stain(stain1, stain2)])
