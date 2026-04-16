from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def get_response(vector_db, user_question):
    """
    Person 4 Task: Connects the Vector DB to Llama 3.2 
    to generate an answer.
    """
    
    llm = Ollama(model="llama3.2:1b", timeout=120) 
    
    #Define the Instruction (System Prompt)
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    # Logic Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2}) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    #  Invoke AI and return answer
    response = retrieval_chain.invoke({"input": user_question})
    return response["answer"]