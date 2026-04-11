from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch
import os
from config import BASE_MODEL_NAME, CHUNKED_JSONL_PATH, TRAIN_OUTPUT_DIR, LOGS_DIR
from utils import format_story_prompt


def main():
    # Настройка путей и устройства
    model_name = BASE_MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Загружаем jsonl
    dataset = load_dataset("json", data_files=CHUNKED_JSONL_PATH)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    def tokenize_function(example):
        text = format_story_prompt(
            example["title"], example["completion"], eos_token=tokenizer.eos_token
        )
        return tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=256,  # Уменьшено для 4GB VRAM
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )

    training_args = TrainingArguments(
        output_dir=TRAIN_OUTPUT_DIR,
        eval_strategy="no",
        overwrite_output_dir=True,
        num_train_epochs=1,  # 1 эпоха для теста
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=100,
        logging_steps=1,
        learning_rate=5e-5,
        fp16=True,
        gradient_checkpointing=True,  # Обязательно для 4GB!
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- Проверка данных перед обучением ---
    # Делаем это внутри main(), чтобы не сломать multiprocessing на Windows
    dl = DataLoader(tokenized_dataset["train"], batch_size=4, collate_fn=data_collator)
    batch = next(iter(dl))
    print("Batch shape verification:", batch["input_ids"].shape)
    # ---------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Final model saving to {TRAIN_OUTPUT_DIR}...")
    trainer.save_model(TRAIN_OUTPUT_DIR)
    print("All done!")


if __name__ == "__main__":
    main()
