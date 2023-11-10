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
USE_CACHE = False
USE_DB = True
DB_IS_LOCAL = True
WORKERS = -1

# Database info
DB_PATH = os.path.join("..", "out", "svm.db")
MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_USER = os.environ.get("MONGO_USER", "root")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "root")
