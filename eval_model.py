import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint_path = "result_model/pythia410-1epochs/checkpoint-248"
checkpoint_path = "/media/vitalii/WDBlue/result_model/pythia410-4epochs-fp16-eval/checkpoint-892"
# 1. Загружаем токенизатор (он не менялся, но лучше брать из той же папки)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# 2. Загружаем саму модель, которую мы учили
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    # torch_dtype=torch.bfloat16,  # Используем bf16 для скорости на 3090
    device_map="cuda"           # Автоматически закинет на GPU
)

model.eval()  # Переводим в режим предсказания
print("Модель успешно загружена из чекпоинта!")

# 3. Пробуем сгенерировать сказку
prompt = "Title: The Dragon and the Little Mouse\n\nIn the galaxy far far away"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.5
    )

print("\n--- РЕЗУЛЬТАТ ОБУЧЕНИЯ ---\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
