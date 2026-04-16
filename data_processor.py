import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdfs(pdf_docs):
    text = ""
    # Extracting text from all uploaded PDFs
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                # Cleaning up extra spaces for better AI accuracy
                text += " ".join(content.split()) + " "

    # Split text into manageable chunks
    # 1000 chars 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    return chunks