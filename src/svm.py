import os
import cv2
import numpy as np
import logging as log
from tqdm import tqdm
from sklearn.svm import SVC
from typing import Callable, Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from skimage.feature import (
    hog,
    local_binary_pattern,
    blob_dog,
    blob_log,
    blob_doh,
    canny,
    daisy,
    haar_like_feature,
    hessian_matrix_det,
    multiblock_lbp,
)

# logging configuration, set level to INFO, format to [YYYY-MM-DD HH:MM:SS] [LEVEL] MESSAGE
log.basicConfig(level=log.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# Data path info
DATA_PATH = os.path.join("..", "data", "a2", "data")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")
FOLDERS = os.listdir(TRAIN_PATH)

# Verbose flag
VERBOSE = True

# Output path info
OUTPUT_PATH = os.path.join("..", "out", "data")

# Other info
USE_CACHE = True
USE_DB = False
if USE_DB:
    import mysql.connector
    from mysql.connector import Error


def load_dataset_info(path: str, verbose: bool = False) -> np.ndarray:
    """Load dataset information from the given path."""
    dataset = []
    for folder in tqdm(os.listdir(path), disable=not verbose, desc="Loading dataset info"):
        for file in os.listdir(os.path.join(path, folder)):
            dataset.append({"label": folder, "img_name": file})
    return np.array(dataset)


def load_dataset(
    path: str, dataset: np.ndarray, img_size=(32, 32), apply_func: Callable = None, func_args: Tuple = (), verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from the given path and dataset information.

    Parameters
    ----------
    path : str
        The path to the dataset.
    dataset : np.ndarray
        The dataset information.
    img_size : tuple, optional
        The size of the image, by default (32, 32).
    apply_func : function, optional
        The function to apply to the image, args=(img), by default None.
    func_args : tuple, optional
        The arguments to pass to the function, by default ().

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The dataset and the labels.
    """
    X = []
    y = []
    desc = "Loading dataset" if apply_func is None else f"Loading dataset using {apply_func.__name__}"
    for data in tqdm(dataset, disable=not verbose, desc=desc):
        img_path = os.path.join(path, data["label"], data["img_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        if apply_func is not None:
            img = apply_func(img, *func_args)
        X.append(img.flatten())
        y.append(data["label"])
    return np.array(X), np.array(y)


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


def extract_blob_features(img, blob_type: str = "dog") -> np.ndarray:
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


def extract_haar_features(img) -> np.ndarray:
    """Extract haar features from the given dataset."""
    return np.array(haar_like_feature(img, r=1, c=1, feature_type="type-2-x", height=25, width=25))


def extract_hessian_features(img) -> np.ndarray:
    """Extract hessian features from the given dataset."""
    return np.array(hessian_matrix_det(img, sigma=1.0))


def extract_multiblock_lbp_features(img) -> np.ndarray:
    """Extract multiblock lbp features from the given dataset."""
    return np.array(multiblock_lbp(img, r=3, c=3, width=8, height=8))


def extract_histogram_features(img) -> np.ndarray:
    """Extract histogram features from the given dataset."""
    return np.array(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten())


if __name__ == "__main__":
    log.info("Loading dataset information...")
    dataset_info = load_dataset_info(TRAIN_PATH, verbose=VERBOSE)
    log.info("Loading dataset...")
    # name, func, func_args
    feautes_map: List[Tuple[str, Callable, Tuple]] = [
        ("hog", extract_hog_features, ()),
        ("lbp", extract_lbp_features, ()),
        ("blob_dog", extract_blob_features, ("dog",)),
        ("blob_log", extract_blob_features, ("log",)),
        ("blob_doh", extract_blob_features, ("doh",)),
        ("canny", extract_canny_features, ()),
        ("daisy", extract_daisy_features, ()),
        ("haar", extract_haar_features, ()),
        ("hessian", extract_hessian_features, ()),
        ("multiblock_lbp", extract_multiblock_lbp_features, ()),
        ("histogram", extract_histogram_features, ()),
    ]
    for name, func, args in feautes_map:
        log.info(f"Extracting {name} features...")
        if USE_CACHE and os.path.exists(os.path.join(OUTPUT_PATH, f"X_{name}.npy")):
            log.info(f"Loading {name} features from cache...")
            X = np.load(os.path.join(OUTPUT_PATH, f"X_{name}.npy"))
            y = np.load(os.path.join(OUTPUT_PATH, f"y_{name}.npy"))
        else:
            X, y = load_dataset(TRAIN_PATH, dataset_info, apply_func=func, func_args=args, verbose=VERBOSE)
            log.info(f"Saving {name} features...")
            np.save(os.path.join(OUTPUT_PATH, f"X_{name}.npy"), X)
            np.save(os.path.join(OUTPUT_PATH, f"y_{name}.npy"), y)

    # TODO:
    # Load datasets and for each dataset:
    #   - Split into train and test
    #   - Scale train and test
    #   - Train SVM, with grid search using different kernels and parameters
    #   - Evaluate on test set
    #   - Save results
