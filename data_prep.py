from matplotlib.pyplot import title
import pandas as pd
import json
from utils import split_example

# Путь к CSV
csv_file = "folk_tales_deduplicated.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Convert dataframe to JSONL format with the required structure


def convert_df_to_jsonl(df, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            title = str(row["title"]).strip()
            story = str(row["text"]).strip()

            # Добавляем пробел в начале completion (LLM-friendly)
            if not story.startswith(" "):
                story = " " + story

            record = {
                "title": title,
                "completion": story
            }

            # 🔑 ВАЖНО: split_example вызывается ОДИН РАЗ
            chunks = split_example(record)

            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


convert_df_to_jsonl(df, "stories_chunked.jsonl")
with open("stories_chunked.jsonl", "r", encoding="utf-8") as f:
    print(len(f.readlines()))
