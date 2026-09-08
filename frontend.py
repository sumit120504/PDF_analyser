import streamlit as st 
from rag_pipeline import answer_query, llm_model
from app import load_pdf, create_chunks, get_embeddings_model
from langchain_community.vectorstores import FAISS
import os

uploaded_file = st.file_uploader("Upload PDF", type = "pdf")
user_query = st.text_area("Ask your question")
ask = st.button("Ask AI")

if ask and uploaded_file and user_query.strip():
    os.makedirs("pdfs", exist_ok=True)
    file_path = "pdfs/" + uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        documents = load_pdf(file_path)
        chunks = create_chunks(documents)
        faiss_db = FAISS.from_documents(chunks, get_embeddings_model())
        retrieved_docs = faiss_db.similarity_search(user_query, k=4)
        response = answer_query(retrieved_docs, llm_model, user_query)
        st.write(response.content)
    except (OSError, ValueError) as error:
        st.error(str(error))
else:
    if ask and not uploaded_file:
        st.warning("Upload a PDF first.")
    elif ask:
        st.warning("Enter a question first.")