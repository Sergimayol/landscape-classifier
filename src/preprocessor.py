import cv2
import numpy as np
from typing import Callable, Literal
from skimage.feature import hog, local_binary_pattern, blob_dog, blob_log, blob_doh, canny, daisy, hessian_matrix_det, multiblock_lbp


def extract_hog_features(img) -> np.ndarray:
    """Extract HOG features from the given image in grayscale."""
    fd = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True,
        feature_vector=True,
    )
    return np.array(fd)


def extract_lbp_features(img) -> np.ndarray:
    """Extract LBP features from the given dataset."""
    return np.array(local_binary_pattern(img, 8, 1, method="uniform"))


def extract_blob_features(img, blob_type: Literal["dog", "log", "doh"] = "dog") -> np.ndarray:
    """Extract blob features from the given dataset."""
    blob_type = blob_type.lower()
    blob_map: dict[str, Callable] = {
        "dog": blob_dog,
        "log": blob_log,
        "doh": blob_doh,
    }
    if blob_type not in blob_map:
        raise ValueError(f"Invalid blob type: {blob_type}")
    blob_func = blob_map[blob_type]
    # length of the feature vector should be 256 if the features dont have size 256 pad with zeros
    res = blob_func(img, max_sigma=30, threshold=0.1)
    if res.shape[0] < 256:
        return np.pad(res, ((0, 256 - res.shape[0]), (0, 0)), mode="constant")
    return np.array(res)


def extract_canny_features(img) -> np.ndarray:
    """Extract canny features from the given dataset."""
    return np.array(canny(img, sigma=3))


def extract_daisy_features(img) -> np.ndarray:
    """Extract daisy features from the given dataset."""
    return np.array(daisy(img, step=180, radius=img.shape[0] // 8, rings=2, histograms=6, orientations=8))


def extract_hessian_features(img) -> np.ndarray:
    """Extract hessian features from the given dataset."""
    return np.array(hessian_matrix_det(img, sigma=1.0))


def extract_multiblock_lbp_features(img) -> np.ndarray:
    """Extract multiblock lbp features from the given dataset."""
    return np.array(multiblock_lbp(img, r=3, c=3, width=8, height=8))


def extract_histogram_features(img) -> np.ndarray:
    """Extract histogram features from the given dataset."""
    return np.array(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten())
