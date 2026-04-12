from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # This must be indented
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

def load_vector_store():
    # Make sure these lines are indented 4 spaces!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # allow_dangerous_deserialization is required for local loading
    vector_db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✅ Database loaded successfully!")
    return vector_db