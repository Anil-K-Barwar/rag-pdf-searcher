# Updated imports for langchain-community
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    """
    Load a PDF and return as a list of Document objects.
    Each Document contains text and metadata.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    # Test the loader with a sample PDF
    docs = load_pdf("data/resume.pdf")
    print(f"✅ Loaded {len(docs)} documents from PDF")
    # Optional: print first 200 characters of first document
    if docs:
        print("Example content:", docs[0].page_content[:200])
