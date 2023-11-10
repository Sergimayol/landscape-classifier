import os
import cv2
import numpy as np
from tqdm import tqdm
import logging as log
from sklearn.svm import SVC
from datetime import datetime
from typing import Callable, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from db import save_to_db
from settings import TRAIN_PATH, TEST_PATH, USE_CACHE, VERBOSE, OUTPUT_PATH, WORKERS
from preprocessor import (
    extract_blob_features,
    extract_canny_features,
    extract_daisy_features,
    extract_hessian_features,
    extract_hog_features,
    extract_histogram_features,
    extract_lbp_features,
    extract_multiblock_lbp_features,
    extract_corner_features,
)

# logging configuration, set level to INFO, format to [YYYY-MM-DD HH:MM:SS] [LEVEL] MESSAGE
log.basicConfig(level=log.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

GLOBAL_DATASET_INFO: dict[str, int] = {}


def load_dataset_info(path: str, except_labels: list[str] = [], verbose: bool = False) -> np.ndarray:
    """Load dataset information from the given path."""
    dataset = []
    for folder in tqdm(os.listdir(path), disable=not verbose, desc="Loading dataset info"):
        if folder.lower() in except_labels:
            continue
        for file in os.listdir(os.path.join(path, folder)):
            dataset.append({"label": folder.lower(), "img_name": file})
    return np.array(dataset)


def pad_or_cut_array(array: np.ndarray, max_len: int) -> np.ndarray:
    """Pad or cut the given array to the given length."""
    if len(array) < max_len:
        return np.pad(array, (0, max_len - len(array)), "constant")
    elif len(array) > max_len:
        return array[:max_len]
    return array


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
    X, y = [], []
    desc = "Loading dataset" if apply_func is None else f"Loading dataset using {apply_func.__name__}"
    len_set = set()
    for data in tqdm(dataset, disable=not verbose, desc=desc):
        img_path = os.path.join(path, data["label"], data["img_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        if apply_func is not None:
            img = apply_func(img, *func_args)
        img = img.flatten()
        len_set.add(len(img))
        X.append(img)
        y.append(data["label"])
    if len(len_set) > 1:
        log.warning(f"Images have different sizes: {len_set}, padding with zeros...")
        max_len = max([len(x) for x in X])
        name = apply_func.__name__ if apply_func is not None else "raw"
        if GLOBAL_DATASET_INFO.get(f"MXL{name}", None) is None:
            GLOBAL_DATASET_INFO[f"MXL{name}"] = max_len
        X = np.array([pad_or_cut_array(x, GLOBAL_DATASET_INFO[f"MXL{name}"]) for x in X])
    return np.array(X), np.array(y)


def get_x_and_y(
    name: str,
    path: str,
    is_train: bool,
    dataset_info: np.ndarray,
    apply_func: Callable,
    func_args: Tuple,
    img_size: Tuple[int, int] = (32, 32),
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get X and y from the given dataset."""
    log.info(f"Extracting {name} features...")
    out_path = "train" if is_train else "test"
    if (
        USE_CACHE
        and os.path.exists(os.path.join(OUTPUT_PATH, out_path, f"X_{name}.npy"))
        and os.path.exists(os.path.join(OUTPUT_PATH, out_path, f"y_{name}.npy"))
    ):
        log.info(f"Loading {name} features from cache...")
        X = np.load(os.path.join(OUTPUT_PATH, out_path, f"X_{name}.npy"))
        y = np.load(os.path.join(OUTPUT_PATH, out_path, f"y_{name}.npy"))
    else:
        X, y = load_dataset(path, dataset_info, img_size, apply_func=apply_func, func_args=func_args, verbose=verbose)
        log.info(f"Saving {name} features...")
        if not os.path.exists(os.path.join(OUTPUT_PATH, out_path)):
            os.makedirs(os.path.join(OUTPUT_PATH, out_path))
        np.save(os.path.join(OUTPUT_PATH, out_path, f"X_{name}.npy"), X)
        np.save(os.path.join(OUTPUT_PATH, out_path, f"y_{name}.npy"), y)
    return X, y


def assert_dataset_labels(dataset_info: np.ndarray, dataset_test_info: np.ndarray) -> None:
    """Assert that the labels of the two datasets are the same."""
    labels_train = [data["label"] for data in dataset_info]
    labels_test = [data["label"] for data in dataset_test_info]
    unique_labels_train = np.unique(labels_train)
    unique_labels_test = np.unique(labels_test)
    assert np.all(unique_labels_train == unique_labels_test) and len(unique_labels_train) == len(
        unique_labels_test
    ), "The labels of the two datasets are not the same."


def reset_global_metadata():
    GLOBAL_DATASET_INFO.clear()


if __name__ == "__main__":
    start_time = datetime.now()
    log.info("Loading dataset information...")
    dataset_info = load_dataset_info(TRAIN_PATH, verbose=VERBOSE)
    dataset_test_info = load_dataset_info(TEST_PATH, except_labels=["livingroom (case conflict)"], verbose=VERBOSE)
    assert_dataset_labels(dataset_info, dataset_test_info)
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
        ("hessian", extract_hessian_features, ()),
        ("multiblock_lbp", extract_multiblock_lbp_features, ()),
        ("histogram", extract_histogram_features, ()),
        ("corner", extract_corner_features, ()),
    ]
    params = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
        "coef0": [0.0, 0.1, 1.0],
        "class_weight": ["balanced", None],
    }
    verb = 1 if VERBOSE else 0
    imgs_sizes = [(28, 28), (32, 32), (64, 64), (128, 128)]
    # for img_size in imgs_sizes:
    for img_size in [(128, 128)]:
        # Train and validation to find best parameters for SVM
        log.info(f"Training and validating using {img_size} image size...")
        for name, func, args in feautes_map:
            X, y = get_x_and_y(name, TRAIN_PATH, True, dataset_info, func, args, img_size, verbose=VERBOSE)
            # Split into train and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            # Scale train and validation
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            log.info(f"Shape of X_train: {X_train.shape}, shape of X_val: {X_val.shape}")
            # Train SVM, with grid search using different kernels and parameters
            log.info(f"Training SVM using {name} features...")
            grid = GridSearchCV(SVC(), params, refit=True, n_jobs=WORKERS, cv=5, verbose=verb)
            grid.fit(X_train, y_train)
            log.info(f"Best parameters for {name} features: {grid.best_params_}, best score: {grid.best_score_}")
            model = grid.best_estimator_
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            log.info(f"Classification report for {name} features:")
            class_rep = classification_report(y_val, y_pred, zero_division=0)
            log.info(class_rep)
            log.info(f"Confusion matrix for {name} features:")
            conf_mat = confusion_matrix(y_val, y_pred)
            log.info(conf_mat)
            # Save info to database
            precision_ = precision_score(y_val, y_pred, average="macro", zero_division=0)
            recall_ = recall_score(y_val, y_pred, average="macro", zero_division=0)
            f1_score_ = f1_score(y_val, y_pred, average="macro", zero_division=0)
            support_ = accuracy_score(y_val, y_pred)
            save_to_db(
                name,
                grid.best_params_,
                grid.best_score_,
                precision_,
                recall_,
                f1_score_,
                support_,
                img_size,
                class_rep,
                conf_mat,
                is_val=True,
            )
            # Ecaluate on test set
            log.info(f"Evaluating on test set using {name} features...")
            X_test, y_test = get_x_and_y(name, TEST_PATH, False, dataset_test_info, func, args, img_size, verbose=VERBOSE)
            # Scale test
            X_test = scaler.transform(X_test)
            log.info(f"Shape of X_test: {X_test.shape} and shape of y_test: {y_test.shape}")
            # Predict
            y_pred = model.predict(X_test)
            # Evaluate
            log.info(f"Classification report for {name} features:")
            class_rep = classification_report(y_test, y_pred, zero_division=0)
            log.info(class_rep)
            log.info(f"Confusion matrix for {name} features:")
            conf_mat = confusion_matrix(y_test, y_pred)
            log.info(conf_mat)
            # Save info to database
            precision_ = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall_ = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1_score_ = f1_score(y_test, y_pred, average="macro", zero_division=0)
            support_ = accuracy_score(y_test, y_pred)
            save_to_db(
                name,
                grid.best_params_,
                grid.best_score_,
                precision_,
                recall_,
                f1_score_,
                support_,
                img_size,
                class_rep,
                conf_mat,
                is_val=False,
            )
            reset_global_metadata()

    end = datetime.now()
    log.info(f"Finished in {end - start_time}")
