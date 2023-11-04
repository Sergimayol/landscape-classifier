import numpy as np
from typing import Tuple
from tinydb import TinyDB
from pymongo import MongoClient
from settings import DB_PATH, USE_DB, DB_IS_LOCAL, MONGO_HOST, MONGO_USER, MONGO_PASSWORD


def __connect_to_local_db():
    return TinyDB(DB_PATH)


def __connect_to_remote_db():
    return MongoClient(MONGO_HOST, connect=True, username=MONGO_USER, password=MONGO_PASSWORD)


def __save_to_local_db(
    name: str,
    best_params: dict,
    best_score: float,
    precision: float,
    recall: float,
    f1_score: float,
    support: float,
    img_size: Tuple[int, int],
    classification_report: str = None,
    confusion_matrix: np.ndarray = None,
    is_val: bool = True,
):
    db = __connect_to_local_db()
    db.insert(
        {
            "name": name,
            "best_params": best_params,
            "best_score": best_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "img_size": img_size,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix.tolist(),
            "is_val": is_val,
        }
    )
    db.close()


def __save_to_remote_db(
    name: str,
    best_params: dict,
    best_score: float,
    precision: float,
    recall: float,
    f1_score: float,
    support: float,
    img_size: Tuple[int, int],
    classification_report: str = None,
    confusion_matrix: np.ndarray = None,
    is_val: bool = True,
):
    client = __connect_to_remote_db()
    db = client["svm"]
    if "svm_train" not in db.list_collection_names():
        db.create_collection("svm_train")
    db = db.get_collection("svm_train")
    db.insert_one(
        {
            "name": name,
            "best_params": best_params,
            "best_score": best_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "img_size": img_size,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix.tolist(),
            "is_val": is_val,
        }
    )
    client.close()


def save_to_db(
    name: str,
    best_params: dict,
    best_score: float,
    precision: float,
    recall: float,
    f1_score: float,
    support: float,
    img_size: Tuple[int, int],
    classification_report: str = None,
    confusion_matrix: np.ndarray = None,
    is_val: bool = True,
):
    """Save the given info to the database."""
    if not USE_DB:
        return

    if DB_IS_LOCAL:
        __save_to_local_db(
            name, best_params, best_score, precision, recall, f1_score, support, img_size, classification_report, confusion_matrix, is_val
        )
    else:
        __save_to_remote_db(
            name, best_params, best_score, precision, recall, f1_score, support, img_size, classification_report, confusion_matrix, is_val
        )
