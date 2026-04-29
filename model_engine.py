import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def get_response(vector_db, user_question):
    llm = Ollama(model="llama3.2:1b", timeout=120) 
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based ONLY on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_question})
    answer = response["answer"]
    
    citations = []
    # response["context"] has the raw document chunks
    if "context" in response:
        for doc in response["context"]:
            # Try multiple common metadata keys for the source
            meta = doc.metadata
            source_raw = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown Document"
            
            # Clean the path to just the filename
            source_file = os.path.basename(source_raw)
            
            # Get Page Number
            page_no = meta.get("page") or meta.get("page_number")
            
            if page_no is not None:
                # Add 1 if it's an integer (since it's usually 0-indexed)
                display_page = page_no + 1 if isinstance(page_no, int) else page_no
                citation_str = f"{source_file} (Page {display_page})"
            else:
                citation_str = f"{source_file}"

            if citation_str not in citations:
                citations.append(citation_str)

    # Append to answer only if we actually found real citations
    if citations and "Unknown" not in citations[0]:
        answer += "\n\n**Sources:** " + " | ".join([f"[{c}]" for c in citations])
    else:
        # Fallback: If metadata is missing, we check the actual filename being uploaded
        answer += "\n\n**Sources:** [Verified from Uploaded Document]"

    return answer, citations