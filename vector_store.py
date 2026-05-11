import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """Creates and saves a local FAISS index with metadata."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

def load_vector_store():
    """Loads the existing FAISS index for session persistence."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None