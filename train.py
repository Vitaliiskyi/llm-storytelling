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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,)

    # Загружаем jsonl
    dataset = load_dataset("json", data_files=CHUNKED_JSONL_PATH)
    dataset = dataset.filter(lambda x: x["token_count"] <= 2000)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    def tokenize_function(example):
        text = format_story_prompt(
            example["title"], example["completion"], eos_token=tokenizer.eos_token
        )
        return tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,

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
        eval_strategy="epoch",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_bnb_8bit",  # вместо "adamw_torch"
        report_to="none",
        warmup_ratio=0.05,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- Проверка данных перед обучением ---
    # Делаем это внутри main(), чтобы не сломать multiprocessing на Windows
    dl = DataLoader(
        tokenized_dataset["train"],
        batch_size=4,
        collate_fn=data_collator
    )
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
