from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

def process_pdf(file_path: str):
    """
    Reads a PDF file from the given path, extracts text, and chunks it.
    Uses RecursiveCharacterTextSplitter to ensure chunks are contextually bounded.
    Returns a list of dictionaries with text and page_number.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Medical texts need reasonably sized chunks to preserve context for graph extraction
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks_data = []
    
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            if chunk.strip():
                chunks_data.append({
                    "text": chunk.strip(),
                    "page_number": page.metadata.get('page', 0)
                })
                
    return chunks_data
