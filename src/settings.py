import os

# Data path info
DATA_PATH = os.path.join("..", "data", "a2", "data")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")
FOLDERS = os.listdir(TRAIN_PATH)

# Verbose flag
VERBOSE = True

# Output path info
OUTPUT_PATH = os.path.join("..", "out")

# Other info
USE_CACHE = True
USE_DB = True
WORKERS = -1

# Database info
DB_PATH = os.path.join("..", "out", "svm.sqlite3")
