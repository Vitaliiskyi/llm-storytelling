# Description: Utility functions for processing storytelling data,
# including splitting long texts into manageable chunks with added noise for variability.

import torch
import random
from nltk.tokenize import sent_tokenize



def format_story_prompt(title: str, completion: str = "", eos_token: str = "") -> str:
    """Форматирует текст сказки для обучения или промпт для генерации."""
    prompt = f"Title: {title}\nStory:\n"

    if completion:
        prompt += completion.strip() + eos_token

    return prompt


# def split_into_chunks(text, tokenizer, max_tokens=1900):
#     sentences = sent_tokenize(text)

#     chunks = []
#     current = []

#     for sent in sentences:
#         candidate = " ".join(current + [sent])
#         token_count = len(
#             tokenizer(candidate, add_special_tokens=False)["input_ids"])

#         if token_count <= max_tokens:
#             current.append(sent)
#         else:
#             if current:
#                 chunks.append(" ".join(current))
#             current = [sent]

#     if current:
#         chunks.append(" ".join(current))

#     return chunks


def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()
