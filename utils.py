import random


TARGET_LEN = 1536
NOISE_MIN = 100
NOISE_MAX = 500
MIN_LAST_CHUNK = 100


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

        results.append({
            "title": title,
            "completion": chunk.strip()
        })

        start = end

    return results
