import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 AI Document Q&A (Mistral)")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded successfully!")
    # This is where Person 2's function will be called