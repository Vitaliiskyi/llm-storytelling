# Description: Utility functions for processing storytelling data,
# including splitting long texts into manageable chunks with added noise for variability.

import torch
import random
TARGET_LEN = 1800
NOISE_MIN = 100
NOISE_MAX = 200
MIN_LAST_CHUNK = 50

def format_story_prompt(title: str, completion: str = "", eos_token: str = "") -> str:
    """Форматирует текст сказки для обучения или промпт для генерации."""
    prompt = f"Title: {title}\n\n"
    if completion:
        prompt += completion + eos_token
    return prompt



def split_example(example,
                  target_len=TARGET_LEN,
                  noise_min=NOISE_MIN,
                  noise_max=NOISE_MAX,
                  min_last_chunk=MIN_LAST_CHUNK):
    title = example["title"]
    text = example["completion"]

    results = []
    start = 0
    text_len = len(text)

    while start < text_len:
        noise = random.randint(noise_min, noise_max)
        chunk_len = target_len + noise
        end = start + chunk_len
        chunk = text[start:end]
        if len(chunk) < min_last_chunk:
            break
        results.append({"title": title, "completion": chunk.strip()})
        start = end

    return results


def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()
