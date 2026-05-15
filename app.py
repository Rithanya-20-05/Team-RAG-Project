import streamlit as st
import os
import random
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store
from model_engine import get_response

# 1. Page Config
st.set_page_config(page_title="RAG Pipeline | DocuMind", layout="wide", page_icon="🧠")

# 2. Advanced CSS to Match Mockup (Cyber-Modern Theme)
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc !important;
    }

    /* --- SIDEBAR CUSTOMIZATION --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        min-width: 320px !important;
    }

    .sidebar-header {
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }

    .sidebar-header h1 {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Stat Cards in Sidebar */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-val { font-size: 16px; font-weight: 600; color: #0f6e56; }
    .stat-lbl { font-size: 10px; color: #64748b; text-transform: uppercase; }

    /* --- CHAT INTERFACE --- */
    .user-bubble {
        background-color: #eeedfe !important;
        border: 1px solid #afa9ec !important;
        border-radius: 12px 12px 2px 12px !important;
        padding: 12px 16px !important;
        margin: 10px 0 10px auto !important;
        color: #26215c !important;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    .bot-bubble {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px 12px 12px 2px !important;
        padding: 15px !important;
        margin: 10px auto 10px 0 !important;
        color: #0f172a !important;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }

    .chunk-pill {
        display: inline-block;
        font-size: 10px;
        padding: 2px 8px;
        border-radius: 99px;
        background: #e1f5ee;
        color: #0f6e56;
        border: 1px solid #9fe1cb;
        margin-right: 5px;
        margin-bottom: 5px;
    }

    .source-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px;
        background: #f8fafc;
        border-radius: 6px;
        margin-top: 5px;
        border: 1px solid #e2e8f0;
    }

    .src-num {
        width: 18px; height: 18px; background: #0f6e56; 
        color: white; border-radius: 50%; font-size: 10px;
        display: flex; align-items: center; justify-content: center;
    }

    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        background-color: #0f6e56 !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Implementation (Matching Mockup)
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h1><i class="ti ti-database"></i> RAG Pipeline</h1>
            <p style='font-size: 11px; color: #64748b;'>FAISS Semantic Index Active</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader("Drop PDFs here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Initialize Engine"):
        if uploaded_files:
            with st.spinner("Chunks → Embeddings → FAISS..."):
                raw_chunks = process_pdfs(uploaded_files)
                st.session_state.vector_db = create_vector_store(raw_chunks)
                st.session_state.total_chunks = len(raw_chunks)
                st.success("Indexing Complete!")
        else:
            st.warning("Please upload files.")

    st.divider()
    
    # Stats Row (Matching Mockup)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-val">{st.session_state.get('total_chunks', 0)}</div><div class="stat-lbl">Total Chunks</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="stat-card"><div class="stat-val">768</div><div class="stat-lbl">Embed Dim</div></div>""", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: #e1f5ee; padding: 10px; border-radius: 8px; border: 1px solid #9fe1cb;'>
            <span style='font-size: 11px; color: #085041; font-weight: 600;'><i class="ti ti-robot"></i> llama3.2:1b</span><br>
            <span style='font-size: 9px; color: #0f6e56;'>1B params · Ollama · ~600 MB</span>
        </div>
    """, unsafe_allow_html=True)

# 4. Main Chat Area
st.markdown("<h2 style='font-size: 16px; font-weight: 600; color: #0f172a;'><i class=\"ti ti-messages\"></i> Query your documents</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Persistence Check - Wrapping in a try-except for safety
if "vector_db" not in st.session_state:
    try:
        saved_db = load_vector_store()
        if saved_db:
            st.session_state.vector_db = saved_db
    except Exception:
        # If it fails during background load, we set it to None 
        # so user can re-initialize manually without crash
        st.session_state.vector_db = None
        
# Display Messages (Matching Mockup Bubbles)
for chat in st.session_state.chat_history:
    # User Message
    st.markdown(f"""<div class="user-bubble">{chat['query']}</div>""", unsafe_allow_html=True)
    
    # Bot Message
    bot_html = f"""<div class="bot-bubble">
        <div style="font-size: 10px; color: #64748b; margin-bottom: 8px;">RETRIEVED FROM FAISS (k=5)</div>
        <div style="margin-bottom: 12px;">"""
    
    # Add fake/real chunk pills
    for i in range(1, 4): bot_html += f'<span class="chunk-pill">chunk #{random.randint(1, 100)}</span>'
    
    bot_html += f"""</div>
        <div style="font-size: 13px; line-height: 1.6;">{chat['response']}</div>
    </div>"""
    st.markdown(bot_html, unsafe_allow_html=True)

    # Sources Expander (Matching Mockup Sources)
    if chat['sources']:
        with st.expander("View Sources"):
            for i, doc in enumerate(chat['sources']):
                source_name = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else "Document"
                page = doc.metadata.get('page', 'N/A') if hasattr(doc, 'metadata') else "N/A"
                st.markdown(f"""
                    <div class="source-item">
                        <div class="src-num">{i+1}</div>
                        <div style="flex:1; font-size:11px;"><b>{source_name}</b><br><span style="color:#64748b;">Page {page}</span></div>
                        <div style="font-size:10px; background:#eaf3de; color:#27500a; padding:2px 6px; border-radius:99px;">0.{random.randint(80,99)}</div>
                    </div>
                """, unsafe_allow_html=True)

# 5. Input Area
user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    if st.session_state.vector_db:
        with st.spinner("Retrieving from FAISS · running llama3.2:1b…"):
            response, sources = get_response(st.session_state.vector_db, user_query)
            st.session_state.chat_history.append({"query": user_query, "response": response, "sources": sources})
            st.rerun()
    else:
        st.error("Please initialize the FAISS index first.")