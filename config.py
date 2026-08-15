import os

# ---- Data Paths ----
RAW_CSV_PATHS = [
    "folk_tales.csv",
    "grimms_fairytales.csv",
]

CHUNKED_JSONL_PATH = "dataset/master_dataset_filtered.jsonl"
# ---- Training Configuration ----

BASE_MODEL_NAME = "EleutherAI/pythia-2.8b"
LOGS_DIR = "./logs"


# Настройка путей в зависимости от операционной системы
if os.name == 'nt':
    # Пути для Windows (тестирование локально)
    TRAIN_OUTPUT_DIR = "./results/pythia-1.4b-output/"
    EVAL_CHECKPOINT_PATH = "./results/pythia-1.4b-output"
else:
    # Пути для Linux / MacOS
    TRAIN_OUTPUT_DIR = "/mnt/WDGreen/llm_storytelling/results/pythia-2.8-2000-1/"
    EVAL_CHECKPOINT_PATH = "/mnt/WDGreen/llm_storytelling/results/pythia-2.8-2000-1/checkpoint-3282"

# Создаем директории, если их нет
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
