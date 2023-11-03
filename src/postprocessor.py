import pickle
import sqlite3
import numpy as np
import pandas as pd
import mysql.connector

from settings import DB_PATH, DB_IS_LOCAL, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER


def __local_postprocessor():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS svm_data_processed;")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS svm_data_processed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            best_params TEXT NOT NULL,
            best_score REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            support REAL NOT NULL,
            img_size TEXT NOT NULL,
            is_val INTEGER NOT NULL
        );"""
    )
    conn.commit()
    # copy data from svm_train to svm_data_processed
    c.execute("INSERT INTO svm_data_processed SELECT * FROM svm_train;")
    conn.commit()
    res = c.execute("SELECT * FROM svm_data_processed WHERE is_val = 0").fetchall()
    for r in res:
        _id = r[0]
        precision = r[4]
        recall = r[5]
        f1_score = r[6]
        support = r[7]
        best_score = (precision + recall + f1_score + support) / 4
        c.execute("UPDATE svm_data_processed SET best_score = ? WHERE id = ?", (best_score, _id))
        conn.commit()
    c.close()
    conn.close()


def __remote_postprocessor():
    conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
    c = conn.cursor()
    c.execute("USE svm;")
    c.execute("DROP TABLE IF EXISTS svm_data_processed;")
    c.execute(
        """
          CREATE TABLE IF NOT EXISTS svm_data_processed (
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
        );"""
    )
    conn.commit()
    # copy data from svm_train to svm_data_processed
    c.execute("INSERT INTO svm_data_processed SELECT * FROM svm_train;")
    conn.commit()
    c.execute("SELECT * FROM svm_data_processed WHERE is_val = 0")
    res = c.fetchall()
    for r in res:
        _id = r[0]
        precision = r[4]
        recall = r[5]
        f1_score = r[6]
        support = r[7]
        best_score = (precision + recall + f1_score + support) / 4
        c.execute("UPDATE svm_data_processed SET best_score = %s WHERE id = %s", (best_score, _id))
        conn.commit()
    c.close()
    conn.close()


if __name__ == "__main__":
    if DB_IS_LOCAL:
        __local_postprocessor()
    else:
        __remote_postprocessor()

    conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
    c = conn.cursor()
    c.execute("USE svm;")
    c.execute("SELECT confusion_matrix FROM svm_data_processed ORDER BY best_score DESC;")
    res = c.fetchall()
    arrs = []
    for r in res:
        arr = pickle.loads(r[0])
        shape = arr.shape
        arr = arr.flatten().tolist()
        arrs.append((arr, shape[0], shape[1]))

    df = pd.DataFrame(arrs, columns=["confusion_matrix", "height", "width"])
    print(df)
    df.to_csv("confusion_matrix.csv", index=False)
