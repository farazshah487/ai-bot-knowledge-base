import streamlit as st
from pdf_processing import extract_text_from_pdf, chunk_text
from url_processing import extract_text_from_url  # <-- new import
from llm import generate_answer
from vector_db import FAISSVectorDB
from openai import OpenAI

client = OpenAI()

st.title("Knowledge Base Bot")

# Select input type
input_type = st.radio("Select input type", ["PDF Upload", "URL"])

user_question = st.text_input("Ask a question:")

text = None
if input_type == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        text = extract_text_from_url(url)

if text and user_question:
    # Split into chunks
    chunks = chunk_text(text)

    # Generate embeddings (updated for new OpenAI client)
    embeddings = [client.embeddings.create(input=chunk, model="text-embedding-3-large").data[0].embedding
                  for chunk in chunks]

    # Store in FAISS
    vector_db = FAISSVectorDB(dimension=len(embeddings[0]))
    vector_db.add_embeddings(embeddings, chunks)

    # Query
    query_embedding = client.embeddings.create(input=user_question, model="text-embedding-3-large").data[0].embedding
    relevant_chunks = vector_db.query(query_embedding, top_k=3)

    # Generate answer
    answer = generate_answer(user_question, relevant_chunks)
    st.subheader("Answer")
    st.write(answer)
