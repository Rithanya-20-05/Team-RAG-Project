import streamlit as st
from data_processor import process_pdfs
from vector_store import create_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 2. Advanced CSS for UI Enhancement (Input Box & Bubbles)
st.markdown("""
    <style>
    /* Chat Input Box styling */
    div[data-testid="stChatInput"] {
        border: 2px solid #4A90E2 !important;
        border-radius: 30px !important;
        padding: 5px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Center titles */
    .stTitle {
        text-align: center;
        color: #2C3E50;
    }

    /* Sidebar buttons */
    .stButton>button {
        border-radius: 20px;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar for Person 1 & 2's work
with st.sidebar:
    st.markdown("## 📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button(" Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Person 2's Logic
                raw_chunks = process_pdfs(uploaded_files)
                # Person 3's Logic
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f" Ready! Created {len(raw_chunks)} chunks.")
                # Task Tracker for 4 members
                st.info("Tasks Completed: Person 1 (UI), Person 2 (Data), Person 3 (DB)")
        else:
            st.warning("Please upload a file first.")

# 4. Main Chat Area
st.title("🤖 Mistral AI Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with clean icons
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input Box (The Enhanced Part)
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 6. Person 4's Integration
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mistral is thinking..."):
                try:
                    answer = get_response(st.session_state.vector_db, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    # Final Task Tracker
                    if len(st.session_state.messages) > 1:
                        st.toast("✅ Person 4: Response Generated!")
                except Exception as e:
                    st.error("Mistral link fail aagudhu! Terminal-la 'ollama run mistral' check pannunga.")
    else:
        st.error("⚠️ Wait for PDFs to be processed. Click 'Process Documents' in the sidebar.")