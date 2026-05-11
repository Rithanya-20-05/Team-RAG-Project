import streamlit as st
import os
import uuid
from data_processor import process_pdfs
from vector_store import create_vector_store, load_vector_store
from model_engine import get_response

def main():
    # 1. Title and Header (As per your template)
    st.title("Document QA Bot")
    st.write("Upload a PDF, initialize the engine, and ask questions!")

    # 2. Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # 3. Sidebar (Using your template structure)
    with st.sidebar:
        st.header("System Status")
        st.info("Using Local Llama 3.2 & FAISS")
        # Persistence Check
        if st.session_state["vector_db"] is None:
            saved_db = load_vector_store()
            if saved_db:
                st.session_state["vector_db"] = saved_db
                st.success("Previous session restored!")

    # 4. Upload PDF document
    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)

    # 5. User Query Input
    user_query = st.text_input("Ask a question based on the document")

    # 6. Process after user submits (Template Logic)
    if st.button("Submit") and uploaded_files and user_query:
        with st.spinner("Processing PDF and Indexing..."):
            try:
                # Process and Create Vector Store
                raw_chunks = process_pdfs(uploaded_files)
                st.session_state["vector_db"] = create_vector_store(raw_chunks)
                
                with st.spinner("Generating response..."):
                    # Get response from local engine
                    response, sources = get_response(st.session_state["vector_db"], user_query)

                    # Save conversation in session state (Matching your template)
                    st.session_state["chat_history"].append({
                        "query": user_query,
                        "response": response,
                        "sources": sources
                    })
            except Exception as e:
                st.error(f"Error: {e}")


    # 7. Display chat history (Template Style)
   # 7. Display chat history (Cleaned View)
    if st.session_state["chat_history"]:
        st.write("---")
        for chat in reversed(st.session_state["chat_history"]):
            st.write(f"**You:** {chat['query']}")
            
            # Bot response-ah direct-ah display pannuvom
            st.write(f"**Bot:** {chat['response']}")
            
            # --- VIEW SOURCES EXPANDER ONLY ---
            if chat['sources']:
                with st.expander("View Sources"):
                    for doc in chat['sources']:
                        # Indha logic source list-ah matum neat-ah kaatum
                        if not isinstance(doc, str) and hasattr(doc, 'metadata'):
                            source_info = doc.metadata.get('source', 'Document')
                            page_info = doc.metadata.get('page', 'N/A')
                            st.write(f"📍 {source_info} (Page {page_info})")
                        else:
                            # Verum string-ah irundha cleaning panni kaatum
                            clean_source = str(doc).split('|')[0].strip('[] ')
                            st.write(f"📍 {clean_source}")
            st.write("---")
if __name__ == "__main__":
    main()