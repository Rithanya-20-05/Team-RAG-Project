import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdfs(pdf_docs):
    """
    Enhanced Processor: Captures metadata (source & page) 
    so the AI can provide accurate citations.
    """
    all_docs = []
    
    # 1. Extract text while preserving metadata
    for pdf in pdf_docs:
        file_name = getattr(pdf, 'name', 'Unknown_File')
        pdf_reader = PdfReader(pdf)
        
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                # Clean the text
                clean_content = " ".join(content.split())
                
                # Create a Document object for each page
                # This stores the text AND the metadata together
                all_docs.append(Document(
                    page_content=clean_content,
                    metadata={
                        "source": file_name,
                        "page": i  # 0-indexed page number
                    }
                ))

    # 2. Split documents into chunks while PRESERVING metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # split_documents automatically passes the metadata from all_docs to the chunks
    final_chunks = text_splitter.split_documents(all_docs)
    
    return final_chunks