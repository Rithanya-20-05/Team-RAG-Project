import streamlit as st
import os
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store # Added load_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 2. Advanced CSS for UI Enhancement
st.markdown("""
    <style>
    div[data-testid="stChatInput"] {
        border: 2px solid #4A90E2 !important;
        border-radius: 30px !important;
        padding: 5px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    .stTitle {
        text-align: center;
        color: #2C3E50;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NEW: PERSISTENCE CHECK (Person 3's Task) ---
# This checks if a database already exists on the disk when the app starts
if "vector_db" not in st.session_state:
    if os.path.exists("faiss_index"):
        try:
            st.session_state.vector_db = load_vector_store()
            st.sidebar.success("Loaded existing Knowledge Base!")
        except Exception as e:
            st.sidebar.error(f"Failed to load existing index: {e}")

# 3. Sidebar
with st.sidebar:
    st.markdown("## 📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Person 2's Logic
                raw_chunks = process_pdfs(uploaded_files)
                # Person 3's Logic
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f"Ready! Created {len(raw_chunks)} chunks.")
              
        else:
            st.warning("Please upload a file first.")

# 4. Main Chat Area
st.title("Llama AI Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 6. Person 4's Integration
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Llama is thinking..."):
                try:
                    answer = get_response(st.session_state.vector_db, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.toast("Person 4: Response Generated!")
                except Exception as e:
                    st.error("Llama connection failed! Make sure 'ollama run llama3.2:3b' is active.")
    else:
        st.error(" Knowledge base not found. Please upload and process a PDF in the sidebar.")