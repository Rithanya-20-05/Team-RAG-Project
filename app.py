import streamlit as st
import os
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="DocuMind Pro", layout="wide", page_icon="🧠")

# 2. UI Styling Overrides
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #fcfcfc; }

    /* --- SIDEBAR CLEANUP: REMOVE TOP GAP --- */
    [data-testid="stSidebarContent"] {
        padding-top: 0rem !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }

    /* Branding Section */
    .brand-container {
        padding: 0.5rem 1rem 1rem 1rem;
        background-color: #ffffff;
    }

    /* Sidebar Cards */
    .sidebar-section {
        background-color: #f8fbff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1effe;
        border-left: 5px solid #3b82f6;
        margin: 5px 15px 15px 15px;
    }

    /* --- CHAT BUBBLES --- */
    .user-bubble {
        background-color: #e0f2fe !important;
        border: 1px solid #7dd3fc;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: #0369a1;
    }
    
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
        font-size: 0.7rem;
        margin-bottom: 4px;
        display: block;
        letter-spacing: 0.5px;
    }

    /* Modern Blue Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100%;
        border: none !important;
        padding: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Implementation
with st.sidebar:
    # Branding Pinned to Top
    st.markdown("""
        <div class="brand-container">
            <h1 style='text-align: left; font-size: 3.2rem; font-weight: 800; margin-bottom: 0px; color: #1e293b; line-height: 1;'>
                DocuMind
            </h1>
            <p style="color: #64748b; font-weight: 600; margin-top: 8px; font-size: 0.95rem;">
                Your Intelligent Research Partner
            </p>
            <p style="color: #22c55e; font-weight: 700; font-size: 0.85rem; margin-top: 15px;">
                ● System Ready
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Upload Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-size:1.1rem; color:#1e40af;'>UPLOAD</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop PDF files here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-size:1.1rem; color:#1e40af;'>PROCESS</h3>", unsafe_allow_html=True)
    if st.button("Start Indexing"):
        if uploaded_files:
            with st.spinner("Processing..."):
                raw_chunks = process_pdfs(uploaded_files)
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.success(f"Indexed {len(raw_chunks)} sections")
        else:
            st.warning("Upload first")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("Llama 3.2 Engine | Local FAISS Store")

    st.divider()
    st.subheader("Session Statistics")
    if 'vector_db' in st.session_state:
        st.success("Index Active")
    else:
        st.warning("No Index Found")
# 4. Persistence Check
if "vector_db" not in st.session_state:
    if os.path.exists("faiss_index"):
        try:
            st.session_state.vector_db = load_vector_store()
            st.sidebar.info("Auto-restored previous session data.")
        except:
            pass

# 5. Main Area
st.title("Intelligent Document Assistant")
st.markdown("Perform semantic search and Q&A on your documents locally")
st.write("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_load" not in st.session_state:
    st.info("Welcome to DocuMind Pro! Upload your PDFs to start researching.")
    st.session_state.first_load = True
# Display Messages (History)
for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
    label = "USER" if message["role"] == "user" else "ASSISTANT"
    
    # Assistant content already includes sources inside the bubble from model_engine
    st.markdown(f"""<div class="{role_class}"><span class="label-text">{label}</span>{message["content"]}</div>""", unsafe_allow_html=True)

# 6. User Input Logic
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""<div class="user-bubble"><span class="label-text">USER</span>{prompt}</div>""", unsafe_allow_html=True)
    
    if "vector_db" in st.session_state:
        with st.spinner("Assistant is typing..."):
            try:
                # Catching Answer (with inline citations) and Sources list from engine
                answer, sources = get_response(st.session_state.vector_db, prompt)
                
                # Show Assistant Bubble - No separate expander, citations are inside 'answer'
                st.markdown(f"""<div class="assistant-bubble"><span class="label-text">ASSISTANT</span>{answer}</div>""", unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Engine Error: {e}")
    else:
        st.error("Please upload and process documents first.")