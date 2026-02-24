# playground with some LLMs

from transformers import AutoModelForCausalLM, AutoTokenizer


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


story_prompt = """Write short children storie (300-500 words) on the theme Kindness and Caring for the Young or Elderly (showing compassion and offering help to those who need it).

Use different characters (for example, children, animals, or fantasy creatures).

Use different settings (school, forest, magical kingdom, etc.).

Story should have its own style and a different ending.

The story must include a clear moral lesson connected to the theme.

The storie should be written for children aged 6–9, easy to understand, warm, and engaging."""


story_prompt = "Deep in the heart of Whispering Woods, there lived a very old fox named "

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"

model_name = "EleutherAI/pythia-2.8b"

# загружаем токенайзер
tokenizer = AutoTokenizer.from_pretrained(model_name)

# загружаем модель (fp16 для экономии памяти)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# inputs = tokenizer(story_prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=1500)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

inputs = tokenizer("Hello, my name is Vitalii!",
                   return_tensors="pt").to(model.device)
tokens = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(tokens[0]))
