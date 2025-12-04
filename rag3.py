import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os
import httpx
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import requests

st.set_page_config(page_title="PDF Q&A RAG App", layout="wide")

st.title("📄 PDF Q&A Chat")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

tiktoken_cache_dir="./token"
os.environ["TIKTOKEN_CACHE_DIR"]=tiktoken_cache_dir
client = httpx.Client(verify=False) 

llm = ChatOpenAI( 
   
) 

embedding_model = OpenAIEmbeddings( 
  
   ) 


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(f"Uploaded: {uploaded_file.name}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    full_text = "\n".join([d.page_content for d in docs])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)

    # embedding_model = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
    vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5})

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                    You are a professional domain expert.

                    Use ONLY the following context to answer the question:
                ------------------
                {context}
                ------------------

                Instructions:
                - Answer accurately and concisely.
                - If the answer is not in the context, say “I do not know”.

                Question: {question}
                Answer:
                """
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    def rag_tool_func(question: str) -> str:
        result = rag_chain.invoke({"query": user_question})
        return result["result"]

    rag_tool = Tool(
        name="PDF_RAG_QA",
        func=rag_tool_func,
        description="You must answer the questions **only using the content of the provided PDF. "
                    "If the answer is not present in the PDF, respond with: The answer is not available in the document."
                    "Do not provide information from outside sources."
    )

    tools=[rag_tool]

    agent = initialize_agent(
        tools=tools,      
        llm=llm,               
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    user_question = st.text_input("Ask anything about the PDF:")

    if user_question:
        response = agent.run(user_question)
        st.subheader("Agent Answer:")
        st.write(response)

    
