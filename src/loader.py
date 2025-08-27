# Updated imports for langchain-community
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdfs(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Loads a PDF file and splits it into text chunks.
    
    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk (in characters).
        chunk_overlap (int): Overlap between chunks for context preservation.

    Returns:
        list: A list of text chunks.
    """
    # Step 1: Load PDF into LangChain document objects
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap,
                            length_function=len
                 )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Test the loader with a sample PDF
    pdf_file = "data/resume.pdf"  # example PDF path
    chunks = load_and_chunk_pdfs(pdf_file)
    print(f"✅ Loaded {len(chunks)} chunks from {pdf_file}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n{'-'*40}")


   