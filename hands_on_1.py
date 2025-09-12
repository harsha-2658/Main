#RAG using Groq

import os

import warnings
import logging

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# os.environ["GROQ_API_KEY"]=""

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

documents=[]
loader=PyPDFLoader("home_loan.pdf")
documents.extend(loader.load())

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
chunks=text_splitter.split_documents(documents)

embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")
db=FAISS.from_documents(chunks,embeddings)
db.save_local("faiss_index")
# db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    # system="You are an Badminton expert. You answer only Badminton related questions from the user provided document(s) only. If not, reply with 'Not found in the documents.'"
    # " If you feel the question is unsafe or out of the context just return them with the message 'cannot provide any answer for this question'  "
)
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold":0.6,"k":4}
    )

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. 
Answer the question **only** using the provided documents. 
If the answer is not in the documents, reply strictly with:
"I couldn’t find anything related to your documents."

Documents:
{context}

Question: {question}
Answer:
"""
)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": custom_prompt
    },
    verbose=True
)


query = "What is the LIV ratio for home loan ?"
# docs = retriever.get_relevant_documents(query)

# if not docs:  
#     print("❌ Sorry, I couldn’t find anything related to your documents.")
# else:
#     result = qa_chain.invoke({"query": query})
#     print("\nAnswer:\n", result["result"])

# query="What is the national bird of India?"
result=qa_chain.invoke({"query":query})
# print(result)
print("\nAnswer:\n", result["result"])
# print("\nSources:")
# for doc in result["source_documents"]:
#     print(" -", doc.metadata.get("source", "<unknown>"))
# while True:
#     query=input("Ask your question : ")
#     if query=="exit" or query=="break":
#         break
#     result=qa_chain.invoke({"query":query})
#     print("\nAnswer:\n", result["result"])
