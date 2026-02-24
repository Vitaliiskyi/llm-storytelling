# import lmstudio as lms

# with lms.Client() as client:
#     model = client.llm.model("qwen/qwen3-4b-2507")
#     result = model.respond("What is the meaning of life?")

#     print(result)

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen/qwen3-4b-2507",
    messages=[{
        "role": "user",
        "content": "Напиши на Python сортировку пузырьком."
    }]
)

print(response.choices[0].message.content)
