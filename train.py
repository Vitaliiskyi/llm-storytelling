from transformers import default_data_collator

from transformers import \
    AutoTokenizer, \
    AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, \
    Trainer, \
    TrainingArguments
from datasets import load_dataset
import torch
import datetime

# model_name = "EleutherAI/pythia-70m"
model_name = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем jsonl
# Набор данных в виде: {"title": "The Thieves and the Cock", "completion": " Some Thieves broke into a house, and ...”"}
dataset = load_dataset("json", data_files="stories_10.jsonl")

# Преобразуем в формат для обучения
# def format_example(example):
#     return {"text": example["title"] + example["completion"]}

MAX_LENGTH = 1024
STRIDE = 256   # overlap (можно 128–512)


def format_example(example):
    return {
        "title": f"Title: {example['title']}\n\n",
        "completion": example["completion"]
    }


dataset = dataset.map(format_example)
dataset = dataset.filter(lambda x: x["title"] and x["completion"])


def tokenize(example):
    title_ids = tokenizer(
        example["title"],
        add_special_tokens=False
    )["input_ids"]

    completion_ids = tokenizer(
        example["completion"],
        add_special_tokens=False
    )["input_ids"]

    ids = title_ids + completion_ids

    return {
        "input_ids": ids,
        "labels": ids.copy()
    }


block_size = 2048


def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // block_size) * block_size

    return {
        "input_ids": [
            concatenated[i:i + block_size]
            for i in range(0, total_len, block_size)
        ],
        "labels": [
            concatenated[i:i + block_size]
            for i in range(0, total_len, block_size)
        ]
    }


tokenized = dataset.map(
    tokenize,
    batched=False,
    remove_columns=dataset["train"].column_names
)

tokenized_dataset = tokenized.map(
    group_texts,
    batched=True
)

# tokenized_dataset = dataset.map(
#     tokenize,
#     batched=False,
#     remove_columns=dataset["train"].column_names,
#     load_from_cache_file=False
# )


training_args = TrainingArguments(
    output_dir="./result_model/pythia410-8epochs " +
    str(datetime.datetime.now()),
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Уменьшено для RTX 3090
    gradient_accumulation_steps=1,   # Компенсируем маленький батч
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    report_to="none",  # отключаем wandb, если не используешь
    # Оптимизации для RTX 3090
    fp16=True,  # Mixed precision для экономии памяти
    dataloader_num_workers=4,  # Параллелизация загрузки данных
    dataloader_pin_memory=True,  # Ускорение загрузки данных
)

# Используем DataCollator для более эффективной обработки
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Для causal language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # eval_dataset=val_dataset,
)


trainer.train()
