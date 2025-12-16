import streamlit as st
from pdf_processing import extract_text_from_pdf, chunk_text
from url_processing import extract_text_from_url
from llm import generate_answer
from vector_db import FAISSVectorDB
from embeddings import create_embeddings
from query_rewriter import rewrite_question
from openai import OpenAI

client = OpenAI()

st.title("Knowledge Base Bot")

# Input selector
input_type = st.radio("Select input type", ["Upload PDF", "URL"])
user_question = st.text_input("Ask a question:")

text = None
if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)

elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        text = extract_text_from_url(url)

if text and user_question:
    with st.spinner("🔍 Reading, indexing, and answering..."):
        # 1️⃣ Chunk text
        chunks = chunk_text(text)

        if not chunks:
            st.error("No readable content found.")
            st.stop()

        # 2️⃣ Generate embeddings (SAFE + throttled)
        embeddings = create_embeddings(chunks)

        if not embeddings:
            st.error("Failed to generate embeddings.")
            st.stop()

        # 3️⃣ Store in FAISS
        vector_db = FAISSVectorDB(dimension=len(embeddings[0]))
        vector_db.add_embeddings(embeddings, chunks[:len(embeddings)])

        # 4️⃣ Embed user question
        # 4️⃣ Rewrite question for better retrieval
        rewritten_question = rewrite_question(
            user_question,
            context_hint=chunks[0]  # first chunk gives topic context
        )

        query_embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=rewritten_question
        ).data[0].embedding

        relevant_chunks = vector_db.query(query_embedding, top_k=3)

        # 5️⃣ Generate answer
        answer = generate_answer(user_question, relevant_chunks)
    st.caption(f"🔎 Interpreted question: {rewritten_question}")
    st.subheader("Answer")
    st.write(answer)

