# RAG-based PDF Searcher

A Retrieval-Augmented Generation (RAG) project to search through PDF documents using embeddings and FAISS.

---

## Project Structure

rag-pdf-searcher/
├── data/     # PDFs to be loaded
├── notebooks/ # experiments and testing
├── src/
│ ├── loader.py    # load PDFs
│ ├── embedder.py    # generate embeddings
│ ├── retriever.py    # query FAISS for relevant chunks
│ └── app.py         # Streamlit UI (to be implemented)
├── README.md
└── requirements.txt

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:Anil-K-Barwar/rag-pdf-searcher.git
cd rag-pdf-searcher
```

### 2. Create a Python virtual environment
```bash
python3 -m venv rag_env

# macOS / Linux
source rag_env/bin/activate

# Windows
#.\rag_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "import langchain_community; import faiss; import sentence_transformers; import pypdf; import streamlit; print('✅ All good!')"
```


