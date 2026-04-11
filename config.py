import os

# ---- Data Paths ----
RAW_CSV_PATH = "folk_tales_deduplicated.csv"
CHUNKED_JSONL_PATH = "stories_chunked.jsonl"

# ---- Training Configuration ----
BASE_MODEL_NAME = "EleutherAI/pythia-160m"
LOGS_DIR = "./logs"

# Пути для Windows (тестирование локально)
TRAIN_OUTPUT_DIR = "./results/pythia160-output/"
EVAL_CHECKPOINT_PATH = "./results/pythia160-output"

# Создаем директории, если их нет
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
