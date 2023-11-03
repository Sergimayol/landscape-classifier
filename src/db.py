import io
import sqlite3
import numpy as np
import mysql.connector
from typing import Tuple
from settings import DB_PATH, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER, USE_DB, DB_IS_LOCAL


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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS svm_train (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL,
            best_params TEXT NOT NULL,
            best_score REAL NOT NULL,
            precision_score REAL NOT NULL,
            recall_score REAL NOT NULL,
            f1_score REAL NOT NULL,
            support_score REAL NOT NULL,
            img_size TEXT NOT NULL,
            classification_report TEXT,
            confusion_matrix LONGBLOB,
            is_val BOOLEAN NOT NULL
        );
        """
    )
    cursor.execute(
        """
        INSERT INTO svm_train (name, best_params, best_score, precision_score, recall_score, f1_score, support_score, img_size, classification_report, confusion_matrix, is_val)
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
    conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS svm;")
    cursor.execute("USE svm;")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS svm_train (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            best_params TEXT NOT NULL,
            best_score REAL NOT NULL,
            precision_score REAL NOT NULL,
            recall_score REAL NOT NULL,
            f1_score REAL NOT NULL,
            support_score REAL NOT NULL,
            img_size TEXT NOT NULL,
            classification_report TEXT,
            confusion_matrix LONGBLOB,
            is_val BOOLEAN NOT NULL
        );
        """
    )
    cursor.execute(
        """
        INSERT INTO svm_train (name, best_params, best_score, precision_score, recall_score, f1_score, support_score, img_size, classification_report, confusion_matrix, is_val)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
            confusion_matrix.dumps(),
            int(is_val),
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()


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
