import random
import matplotlib.pyplot as plt
from dataclasses import dataclass

from settings import DB_IS_LOCAL

if DB_IS_LOCAL:
    from db import __connect_to_local_db
else:
    from db import __connect_to_remote_db


@dataclass
class Data:
    _id: str
    name: str
    best_params: dict
    best_score: float
    precision: float
    recall: float
    f1_score: float
    support: float
    img_size: tuple
    classification_report: str
    confusion_matrix: list[list[int]]
    is_val: bool


def plot_val_vs_test(docs: list[Data]):
    random.seed(42)
    # Plot the best_score for each document on a bar chart
    data = [(doc.name, doc.best_score) for doc in docs]
    x, y = zip(*data)
    _, ax = plt.subplots()
    bars = ax.barh(x, y)
    for i in range(0, len(bars), 2):
        rand_color = "#%06x" % random.randint(0, 0xFFFFFF)
        bars[i].set_color(rand_color)
        bars[i + 1].set_color(rand_color)
    ax.set_xlabel("Best Score")
    ax.set_ylabel("Model")
    ax.set_title("Best Score for each model (val vs test)")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if DB_IS_LOCAL:
        db = __connect_to_local_db()
    else:
        client = __connect_to_remote_db()
        db = client["svm"]

    # Get the collection
    collection = db["svm_train"]

    # Get all the documents
    documents = collection.find()
    docs = [Data(**doc) for doc in documents]

    # For each document is_val is False, set the best_score to the avg of precision, recall, f1_score and support
    for doc in docs:
        if not doc.is_val:
            doc.best_score = (doc.precision + doc.recall + doc.f1_score + doc.support) / 4
            doc.name += " (train)"
        else:
            doc.name += " (val)"

    plot_val_vs_test(docs)

    # Close the connection
    if DB_IS_LOCAL:
        db.close()
    else:
        client.close()
