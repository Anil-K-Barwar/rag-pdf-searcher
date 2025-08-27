# src/retriever.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
OUTPUT_DIR = "outputs"

def run_query(query_text: str, k: int = 3):
    # Load embeddings and FAISS index
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    faiss_index = FAISS.load_local(
        OUTPUT_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Perform similarity search with scores
    results_with_scores = faiss_index.similarity_search_with_score(query_text, k=k)
    return results_with_scores

if __name__ == "__main__":
    # query_text = "What are the key skills highlighted in the resume?"
    query_text = "List major projects completed by the candidate."
    results_with_scores = run_query(query_text)
    print(f'Query :- : {query_text}')
    
    print("\nTop results:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\nResult {i}:")
        print(f"Score: {score:.4f}")
        print(doc.page_content[:500])  # show first 500 characters
        print("-" * 40)
