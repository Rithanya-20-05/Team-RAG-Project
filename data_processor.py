from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdfs(pdf_docs):
    # Logic to load PDF and split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for pdf in pdf_docs:
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        for page in pages:
            chunks.extend(text_splitter.split_text(page.page_content))
    return chunks