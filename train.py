from torch.utils.data import DataLoader
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
from config import BASE_MODEL_NAME, CHUNKED_JSONL_PATH, TRAIN_OUTPUT_DIR, LOGS_DIR
from utils import format_story_prompt

model_name = BASE_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем jsonl
# Набор данных в виде: {"title": "The Thieves and the Cock", "completion": " Some Thieves broke into a house, and ...”"}
dataset = load_dataset("json", data_files=CHUNKED_JSONL_PATH)

# def tokenize(example):
#     text = f"Title: {example['title']}\n\n{example['completion']}"
#     ids = tokenizer(text, add_special_tokens=False)["input_ids"]
#     return {
#         "input_ids": ids,
#         "labels": ids.copy()
#     }
dataset = dataset["train"].train_test_split(
    test_size=0.2,
    seed=42
)


def tokenize(example):
    text = format_story_prompt(example['title'], example['completion'], eos_token=tokenizer.eos_token)

    # Возвращаем input_ids и attention_mask.
    # Обрезаем только экстремально длинные тексты, которые не влезут в Pythia.
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=2048  # Максимальный контекст Pythia
    )


tokenized_dataset = dataset.map(
    tokenize,
    batched=False,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False
)


training_args = TrainingArguments(
    output_dir=TRAIN_OUTPUT_DIR,
    # /media/vitalii/WDBlue/result_model/pythia410-4epochs/
    eval_strategy="steps",
    eval_steps=50,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=2,  # Уменьшено для RTX 3090
    gradient_accumulation_steps=16,   # Компенсируем маленький батч
    save_steps=500,
    save_total_limit=2,
    logging_dir=LOGS_DIR,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,

    # Оптимизации для RTX 3090
    fp16=True,  # Mixed precision для экономии памяти
    dataloader_num_workers=2,  # Параллелизация загрузки данных
    dataloader_pin_memory=True,  # Ускорение загрузки данных
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="tensorboard",          # включаем поддержку TensorBoard
    weight_decay=0.01

)

# Используем DataCollator для более эффективной обработки
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Для causal language modeling
)


dl = DataLoader(
    tokenized_dataset["train"],
    batch_size=4,
    collate_fn=data_collator
)

batch = next(iter(dl))
print("Batch shape:", batch["input_ids"].shape)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    eval_dataset=tokenized_dataset["test"],
)

# Автоматически ищет последнюю папку checkpoint в output_dir
# trainer.train(resume_from_checkpoint=True)
trainer.train()
