from openai import OpenAI
import time

client = OpenAI()

def create_embeddings(chunks, model="text-embedding-3-large"):
    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(
                model=model,
                input=chunk[:8000]  # ✅ hard safety limit
            )
            embeddings.append(response.data[0].embedding)
            time.sleep(0.1)  # ✅ prevents rate limit spikes

        except Exception as e:
            print(f"Embedding failed at chunk {i}: {e}")

    return embeddings
