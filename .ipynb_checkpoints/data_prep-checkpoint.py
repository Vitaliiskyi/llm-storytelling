import pandas as pd
import json

# Путь к CSV
csv_file = "folk_tales_deduplicated.csv"

# Названия колонок в CSV
title_col = "title"       # колонка с названием
story_col = "text"       # колонка с текстом сказки

# Читаем CSV
df = pd.read_csv(csv_file)

# Открываем файл JSONL для записи
with open("stories_10.jsonl", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        title = str(row[title_col]).strip()
        story = str(row[story_col]).strip()

        # Добавляем пробел перед текстом completion
        if not story.startswith(" "):
            story = " " + story

        record = {
            "title": f"{title}",
            "completion": story
        }

        # Записываем как одну строку JSON
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Готово! JSONL сохранён как stories_10.jsonl с {len(df)} записями.")
