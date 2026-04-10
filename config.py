# ---- Data Paths ----
# Путь к исходному скачанному CSV-файлу с датасетом.
RAW_CSV_PATH = "folk_tales_deduplicated.csv"

# Путь к обработанному датасету (результат работы data_prep.py), который подается в модель.
CHUNKED_JSONL_PATH = "stories_chunked.jsonl"

# ---- Training Configuration ----
# Название базовой архитектуры/модели на HuggingFace 
BASE_MODEL_NAME = "EleutherAI/pythia-160m"  # или "EleutherAI/pythia-410m"

# Путь к директории для логов TensorBoard
LOGS_DIR = "./logs"

# Путь к директории для сохранения результатов обучения (веса модели, чекпоинты)
TRAIN_OUTPUT_DIR = "/media/vitalii/WDBlue/result_model/pythia160-output/"

# ---- Evaluation Configuration ----
# Путь к конкретному чекпоинту для генерации сказок и оценки (замените на нужный перед запуском eval)
EVAL_CHECKPOINT_PATH = "/media/vitalii/WDBlue/result_model/pythia410-4epochs-fp16-eval/checkpoint-892"
