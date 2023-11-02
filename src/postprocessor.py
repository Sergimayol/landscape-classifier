from settings import IS_LOCAL, DB_PATH

if IS_LOCAL:
    import sqlite3
else:
    import mysql.connector


if __name__ == "__main__":
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
        id = r[0]
        precision = r[4]
        recall = r[5]
        f1_score = r[6]
        support = r[7]
        best_score = (precision + recall + f1_score + support) / 4
        c.execute("UPDATE svm_data_processed SET best_score = ? WHERE id = ?", (best_score, id))
        conn.commit()
    conn.close()
