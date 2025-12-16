from openai import OpenAI

client = OpenAI()


def generate_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = (
        f"Use the following resume details to answer the question.\n\n"
        f"Resume context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
