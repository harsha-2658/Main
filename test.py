import os
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentType, initialize_agent, Tool
# from langchain_groq import ChatGroq
# from langchain_community.tools.llm_math.tool import LLMMathTool



documents=[]

loader=WebBaseLoader("https://www.yonex.com/badminton/racquets/nanoflare/nf-1000z")
docs=loader.load()

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=150).split_documents(docs)
# chunks=text_splitter.split(docs)

# embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")

# db=FAISS.from_documents(chunks,embeddings)
db=FAISS.from_documents(RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=150).split_documents(docs),OllamaEmbeddings(model="nomic-embed-text:latest"))

llm=ChatOllama(model="llama3:latest")
# llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

retriever=db.as_retriever()

# first_tool=create_retriever_tool(retriever,name="first",description="You are agent who answers the questions from the provided website. If not found strictly reply ' Not found '")
first_tool=Tool(
    name="WebsiteInfo",
    description="You are agent and only answers the questions based on the provided website",
    func=lambda q: retriever.get_relevant_documents(q)

)

# second_tool=LLMMathTool()

agent=initialize_agent(
    tools=[first_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=30,
    max_execution_time=60
)

query="Give me the details about Nanoflare 1000z badminton racket"
response=agent.run(query)
print(response)
# print(first_tool.name)