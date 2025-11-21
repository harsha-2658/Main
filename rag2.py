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


st.set_page_config(page_title="PDF Q&A RAG App", layout="wide")

st.title("📄 PDF Q&A Chat")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

tiktoken_cache_dir="./token"
os.environ["TIKTOKEN_CACHE_DIR"]=tiktoken_cache_dir
client = httpx.Client(verify=False) 

llm = ChatOpenAI( 
   base_url="https://genailab.tcs.in", 
   model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
   api_key="sk-scLhBXHvk9qrq4SX6NFzdA", 
   http_client=client 
) 

embedding_model = OpenAIEmbeddings( 
   base_url="https://genailab.tcs.in", 
   model="azure/genailab-maas-text-embedding-3-large", 
   api_key="sk-scLhBXHvk9qrq4SX6NFzdA", 
   http_client=client
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

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        result = rag_chain.invoke({"query": user_question})
        st.subheader("Answer:")
        st.write(result["result"])

        st.subheader("Source Documents:")
        for doc in result["source_documents"]:
            st.write(doc.metadata)

    os.remove(pdf_path)
