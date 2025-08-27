# src/embedder.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from loader import load_and_chunk_pdfs

import os
import pickle
import faiss

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_faiss_index(file_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load PDF, chunk it, embed chunks, and build FAISS index.

    Args:
        file_path (str): Path to the PDF file.
        model_name (str): Name of HuggingFace embedding model.

    Returns:
        FAISS: A FAISS vectorstore object containing chunk embeddings.
    """
    # Step 1: Load and chunk the PDF
    chunks = load_and_chunk_pdfs(file_path)

    # Step 2: Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Step 3: Create FAISS index from chunks
    faiss_index = FAISS.from_documents(chunks, embeddings)

    return faiss_index


if __name__ == "__main__":
    pdf_path = "data/resume.pdf"
    index = build_faiss_index(pdf_path)
    print("✅ FAISS index built successfully!")
    print(f"Number of vectors stored: {index.index.ntotal}")

    # Save FAISS index and chunks using LangChain's API
    index.save_local(OUTPUT_DIR)
    print(f"FAISS index saved to {OUTPUT_DIR}")


