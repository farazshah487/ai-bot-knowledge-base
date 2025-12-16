from openai import OpenAI

client = OpenAI()

def rewrite_question(question, context_hint):
    prompt = f"""
Rewrite the following question so it is clear, specific,
and suitable for searching the document.

Original question:
{question}

Document context:
{context_hint}

Rewritten question:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite vague questions into clear search queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
