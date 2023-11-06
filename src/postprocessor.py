import os
import random
import seaborn as sns
from tqdm import tqdm
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
    img_size: tuple[int, int]
    classification_report: str
    confusion_matrix: list[list[int]]
    is_val: bool


def plot_val_vs_test(docs: list[Data]):
    random.seed(42)
    X_28, Y_28 = [], []
    X_32, Y_32 = [], []
    X_64, Y_64 = [], []
    X_128, Y_128 = [], []
    for doc in docs:
        if doc.img_size[0] == 28:
            X_28.append(doc.best_score)
            Y_28.append(doc.name)
        elif doc.img_size[0] == 32:
            X_32.append(doc.best_score)
            Y_32.append(doc.name)
        elif doc.img_size[0] == 64:
            X_64.append(doc.best_score)
            Y_64.append(doc.name)
        elif doc.img_size[0] == 128:
            X_128.append(doc.best_score)
            Y_128.append(doc.name)
    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax in axs.flat:
        ax.set(xlabel="best_score", ylabel="name")

    bars = []
    for i, ax in enumerate(axs.ravel()):
        if i == 0:
            bh = ax.barh(Y_28, X_28)
            bars.append(bh)
            ax.set_title("28x28")
        elif i == 1:
            bh = ax.barh(Y_32, X_32)
            bars.append(bh)
            ax.set_title("32x32")
        elif i == 2:
            bh = ax.barh(Y_64, X_64)
            bars.append(bh)
            ax.set_title("64x64")
        elif i == 3:
            bh = ax.barh(Y_128, X_128)
            bars.append(bh)
            ax.set_title("128x128")
        ax.set_xlabel("Best Score")
        ax.set_ylabel("Model")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.invert_yaxis()
        ax.bar_label(bh, fmt="%.3f")

    for bar in bars:
        for i in range(0, len(bar), 2):
            rand_color = "#%06x" % random.randint(0, 0xFFFFFF)
            bar[i].set_color(rand_color)
            bar[i + 1].set_color(rand_color)

    plt.tight_layout()
    plt.show()


def generate_conf_mat(docs: list[Data]):
    path = os.path.join("..", "out", "conf_mat")
    if not os.path.exists(path):
        os.makedirs(path)

    for doc in tqdm(docs, desc="Generating confusion matrices"):
        if doc.is_val:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(doc.confusion_matrix, annot=True, ax=ax, fmt="d")
            ax.set_title(doc.name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{doc.name}.png"))
            plt.close(fig)


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
            doc.name += " (test)"
        else:
            doc.name += " (val)"

    # Generate the confusion matrices
    generate_conf_mat(docs)

    # Plot the best_score for each model
    plot_val_vs_test(docs)

    # Close the connection
    if DB_IS_LOCAL:
        db.close()
    else:
        client.close()
