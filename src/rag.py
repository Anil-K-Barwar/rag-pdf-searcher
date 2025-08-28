# src/rag.py

import os
from retriever import run_query
from dotenv import load_dotenv
from openai import OpenAI

import warnings
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress HuggingFace parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

# Load variables from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def rag_query(user_query: str, k: int = 3):
    # Step 1: Retrieve relevant chunks
    chunks = run_query(user_query, k)

    # context = "\n\n".join([doc.page_content for doc in chunks])
    context = "\n\n".join([doc.page_content for doc, score in chunks])


    # Step 2: Build prompt
    prompt = f"""
    You are a helpful assistant.
    Use the following context from the documents to answer the question.
    If the answer is not in the context, just say "Not found in document".

    Context:
    {context}

    Question: {user_query}
    Answer:
    """

    # Step 3: Ask LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    q = "What are the key skills/Tools highlighted in the resume?"
    answer = rag_query(q)
    print("\nAnswer:\n", answer)
