import streamlit as st
import os
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="DocuMind Pro", layout="wide")

# 2. Enhanced UI Styling with Color-Coded Chat
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #fcfcfc; }

    /* --- Sidebar Enhancement --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }
    
    /* Blue Sidebar Card Effect */
    .sidebar-section {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }

    /* --- Chat Message Overrides --- */
    /* User Message - Blue Theme */
    .user-bubble {
        background-color: #e0f2fe !important;
        border: 1px solid #7dd3fc;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: #0369a1;
    }
    
    /* Assistant Message - Green Theme */
    .assistant-bubble {
        background-color: #f0fdf4 !important;
        border: 1px solid #86efac;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: #166534;
    }

    .label-text {
        font-weight: 800;
        text-transform: uppercase;
        font-size: 0.75rem;
        margin-bottom: 5px;
        display: block;
    }

    /* Button Styling */
    .stButton>button {
        background: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100%;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    # Custom HTML for Bold and Big Branding
    st.markdown("""
        <h1 style='text-align: left; font-size: 2.8rem; font-weight: 800; margin-bottom: 0px;'>
            DocuMind
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown('<p style="color: #22c55e; font-weight: bold; margin-top: 0px;">● Your Intelligent Research Partner</p>', unsafe_allow_html=True)
    st.divider()
    # Step 1 Section with Blue Card
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2 Section with Blue Card
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### process & index")
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                raw_chunks = process_pdfs(uploaded_files)
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f"Indexed {len(raw_chunks)} sections")
        else:
            st.warning("Upload first")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.info("Llama 3.2 Engine | Local FAISS")

# 4. Main Area
st.title("Intelligent Document Assistant")
st.markdown("Perform semantic search and Q&A on your documents locally")
st.write("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages with Custom Bubbles (No Icons)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""<div class="user-bubble"><span class="label-text">USER</span>{message["content"]}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="assistant-bubble"><span class="label-text">ASSISTANT</span>{message["content"]}</div>""", unsafe_allow_html=True)

# 5. User Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""<div class="user-bubble"><span class="label-text">USER</span>{prompt}</div>""", unsafe_allow_html=True)
    
    if "vector_db" in st.session_state:
        with st.spinner("Assistant is typing..."):
            try:
                answer = get_response(st.session_state.vector_db, prompt)
                st.markdown(f"""<div class="assistant-bubble"><span class="label-text">ASSISTANT</span>{answer}</div>""", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except:
                st.error("Engine Error")
    else:
        st.error("Please upload documents.")