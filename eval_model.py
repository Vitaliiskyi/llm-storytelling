import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import EVAL_CHECKPOINT_PATH
from utils import format_story_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Генерация сказки с помощью обученной модели."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="The Dragon and the Little Mouse",
        help="Название сказки",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help="Начальная фраза для генерации (опционально)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Сколько токенов сгенерировать"
    )
    args = parser.parse_args()

    checkpoint_path = EVAL_CHECKPOINT_PATH

    # 1. Загружаем токенизатор (он не менялся, но лучше брать из той же папки)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # 2. Загружаем саму модель, которую мы учили
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        # torch_dtype=torch.bfloat16,  # Используем bf16 для скорости на 3090
        device_map="cuda",  # Автоматически закинет на GPU
    )

    model.eval()  # Переводим в режим предсказания
    print(f"Модель успешно загружена из чекпоинта: {checkpoint_path}")

    # 3. Пробуем сгенерировать сказку
    prompt = format_story_prompt(args.title, args.start)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(
        f"\nНачинаем генерацию сказки...\nНазвание: '{args.title}'\nНачало: '{args.start}'"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.5,
        )

    print("\n--- РЕЗУЛЬТАТ ОБУЧЕНИЯ ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n" + "-" * 30)


if __name__ == "__main__":
    main()
