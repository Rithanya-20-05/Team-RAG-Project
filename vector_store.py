from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # 1. The Embedding Model (This is the 'Encoder')
    # This turns your text chunks into numbers (vectors)
    # We use a free, lightweight model from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. The Vector Database (The 'Storage')
    # This creates a FAISS index which is super fast for searching
    vector_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    # 3. Persistence
    # This saves the DB as a folder so Person 4 can use it easily
    vector_db.save_local("faiss_index")
    
    print("Vector Store created and saved locally!")
    return vector_db