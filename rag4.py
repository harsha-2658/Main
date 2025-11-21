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
import io
import sys
from googlesearch import search
from langchain.agents import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults



st.set_page_config(page_title="Multi-Agent", layout="wide")

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


with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    pdf_path = tmp_file.name
st.success(f"Uploaded: {uploaded_file.name}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

full_text = "\n".join([d.page_content for d in docs])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(full_text)

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
    result = rag_chain.invoke({"query": question})
    return result["result"]

rag_tool = Tool(
        name="PDF_RAG_QA",
        func=rag_tool_func,
        description="Use this tool ONLY when the question is clearly about the content of the PDF. "
                    "Do NOT use this tool for general knowledge or questions unrelated to the PDF."
    )


# duck_search = DuckDuckGoSearchResults()
# from googlesearch import search

# for url in search("latest AI news", num_results=5):
#     print(url)


# web_search_tool_obj = Tool(
#     name="WebSearch",
#     func=duck_search.run,
#     description="Use this tool to get the latest information from the web."
# )
# print(duck_search.run("latest AI news"))


def google_search_tool(query: str) -> str:
    """
    Perform a Google search and return top 5 results as a string.
    """
    try:
        results = list(search(query, num_results=5))
        if results:
            return "\n".join(results)
        else:
            return "No results found."
    except Exception as e:
        return f"Error during search: {e}"


google_search_tool_obj = Tool(
    name="GoogleSearch",
    func=google_search_tool,
    description="Use this tool when the answer cannot be found in the PDF or when the question"
                "is about the real world, news, general knowledge, or anything outside the PDF."
)
# web_search_tool_obj = Tool(
#         name="WebSearch",
#         func=web_search_tool,
#         description="Use this tool to get the latest information from the web."
    
#     )

# tools=[rag_tool,web_search_tool_obj]

tools=[rag_tool,google_search_tool_obj]


supervisor_agent = initialize_agent(
        tools=tools,      
        llm=llm,               
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

user_query = st.text_input("Ask anything about the PDF:")
answer = supervisor_agent.invoke(user_query)
st.write(f"Answer: {answer}")
    # if user_question:
    #     response = supervisor_agent.run(user_question)
    #     st.subheader("Agent Answer:")
    #     st.write(response)
# if st.button("Ask"):
#     if user_query.strip() == "":
#         st.warning("Please enter a question!")
#     else:
#         # Capture verbose output

#         buffer = io.StringIO()
#         sys.stdout = buffer  # Redirect stdout to capture verbose tool calls

#         answer = supervisor_agent.invoke(user_query)

#         sys.stdout = sys.__stdout__  # Reset stdout
#         log = buffer.getvalue()

#         # Extract the tool used from the verbose log (approximate)
#         if "RAG" in log:
#             tool_used = "RAG"
#         elif "WebSearch" in log:
#             tool_used = "WebSearch"
#         else:
#             tool_used = "Unknown"

#         st.success(f"Tool Used: {tool_used}")
#         st.write(f"Answer: {answer}")
#         st.text("Verbose log (for debugging):")
#         st.text(log)
