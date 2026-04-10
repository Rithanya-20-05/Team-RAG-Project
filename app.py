import streamlit as st
from data_processor import process_pdfs
# Importing from the other team members' files
from vector_store import create_vector_store
from model_engine import get_response

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Sidebar for Person 1 & 2's work
with st.sidebar:
    st.title("📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Person 2's Logic
                raw_chunks = process_pdfs(uploaded_files)
                # Person 3's Logic (Initializing the DB)
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f"Ready! Created {len(raw_chunks)} chunks.")
        else:
            st.warning("Please upload a file first.")

# Main Chat Area for Person 4's work
st.title("🤖 Mistral AI Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Person 4's Logic will happen here
    if "vector_db" in st.session_state:
        answer = get_response(st.session_state.vector_db, prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.error("Please process documents in the sidebar first!")