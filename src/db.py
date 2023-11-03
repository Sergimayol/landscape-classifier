import io
import sqlite3
import numpy as np
from typing import Tuple
from settings import DB_PATH, USE_DB


def adapt_array(arr) -> sqlite3.Binary:
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text) -> np.ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def allow_numpy():
    if USE_DB:
        # https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("nparray", convert_array)


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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS svm_train (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            best_params TEXT NOT NULL,
            best_score REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            support REAL NOT NULL,
            img_size TEXT NOT NULL,
            classification_report TEXT,
            confusion_matrix nparray,
            is_val INTEGER NOT NULL
        );
        """
    )
    cursor.execute(
        """
        INSERT INTO svm_train (name, best_params, best_score, precision, recall, f1_score, support, img_size, classification_report, is_val)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            name,
            str(best_params),
            best_score,
            precision,
            recall,
            f1_score,
            support,
            str(img_size),
            str(classification_report),
            confusion_matrix,
            int(is_val),
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
