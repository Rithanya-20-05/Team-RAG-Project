import streamlit as st
import os
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="DocuMind Assistant", layout="wide")

# 2. User-Convenient Professional Light Theme
st.markdown("""
    <style>
    /* Main Background - Soft & Easy on eyes */
    .stApp {
        background-color: #fcfcfc;
        color: #2d3436;
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #edf2f7;
    }

    /* Titles - Bold & Clear */
    h1 {
        color: #1a202c !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Chat Bubbles - Modern & Distinct */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem !important;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* User Message - Slight Tint for Convenience */
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: #f7fafc !important;
    }

    /* Buttons - Large & Accessible */
    .stButton>button {
        background-color: #3182ce !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2b6cb0 !important;
        transform: translateY(-1px);
    }

    /* Chat Input - Fixed at Bottom */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid #cbd5e0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Simplified for Convenience
with st.sidebar:
    st.markdown("# 🧠 DocuMind")
    st.markdown('<p style="color: #38a169; font-weight: 600; font-size: 0.9rem;">🟢 System Ready</p>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📤 Step 1: Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, help="You can upload multiple research papers here.")
    
    st.markdown("### ⚙️ Step 2: Setup")
    if st.button("Process & Learn"):
        if uploaded_files:
            with st.spinner("Building your knowledge base..."):
                raw_chunks = process_pdfs(uploaded_files)
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f"Indexed {len(raw_chunks)} sections!")
        else:
            st.warning("Please upload a PDF first.")

    st.divider()
    st.caption("v1.0 | Local Llama 3.2 Engine")

# 4. Persistence Check
if "vector_db" not in st.session_state:
    if os.path.exists("faiss_index"):
        try:
            st.session_state.vector_db = load_vector_store()
            st.sidebar.info("💡 Auto-loaded previous session data.")
        except:
            pass

# 5. Main Area - Clean & Spaced
st.title("Intelligent Document Assistant")
st.write("Start a conversation with your documents. The AI will answer based on the provided context.")
st.divider()

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I have analyzed your documents. How can I help you today?"}]

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 6. User Input Logic
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    answer = get_response(st.session_state.vector_db, prompt)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except:
                    st.error("Connection Error: Please check if Ollama is running.")
    else:
        st.error("Please upload and process a document in the sidebar to start.")