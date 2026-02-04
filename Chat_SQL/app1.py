import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain : chat with sql DB")
st.title("Langchain : Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# radio_opt = ["Use SQLLite 3 Database - Student.db" , "Connect to you SQL Database"]
radio_opt = ["Use SQLLite 3 Database - Student.db" , "Connect to you SQL Database"]


select_opt = st.sidebar.radio(label="Choose the DB which you want to chat : ", options=radio_opt)

if radio_opt.index(select_opt) == 1:
    db_url = MYSQL
    mysql_host = st.sidebar.text_input("Provide My SQL host name ")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("Mysql Password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_url=LOCALDB

# api_key = st.sidebar.text_input("Enter GROQ API Key" , type="password")

if not db_url:
    st.info("Please enter the database information and url")

# if not api_key:
#     st.info("Please enter API kay")

#LLM Model

# llm = ChatGroq(groq_api_key = api_key , model_name = "openai/gpt-oss-20b" , streaming=True)
# llm = ChatGroq(
#     groq_api_key=api_key,
#     model="openai/gpt-oss-20b",   # or llama3-8b-8192
#     temperature=0,
#     streaming=True,
#     tool_choice="auto"        # 🔑 REQUIRED FOR SQL AGENTS
# )

# llm = ChatGroq(
#     groq_api_key=api_key,
#     model="openai/gpt-oss-20b",
#     temperature=0,
#     streaming=True,
#     model_kwargs={
#         "tool_choice": "auto"
#     }
# )

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


@st.cache_resource(ttl="2h")
def configure_db(db_url, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_url == LOCALDB:
        dbfilepath = Path(__file__).parent / "student.db"

        if not dbfilepath.exists():
            st.error(f"Database file not found at {dbfilepath}")
            st.stop()

        engine = create_engine(f"sqlite:///{dbfilepath}")
        return SQLDatabase(engine)

    elif db_url == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
        return SQLDatabase(engine)

if db_url == MYSQL:
    db= configure_db(db_url , mysql_host , mysql_user , mysql_password ,mysql_db)
else:
    db = configure_db(db_url)

toolkit = SQLDatabaseToolkit(db=db ,llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role" : "assistant" , "content" : "How can I help you ?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anythuing from the database")

if user_query:
    st.session_state.messages.append({"role" : "user" , "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        # response = agent.run(user_query , callbacks=[streamlit_callback])
        response = agent.invoke({"input": user_query},callbacks=[streamlit_callback])
        response_text = response["output"]

        # st.session_state.messages.append({"role" : "assistant" , "content" : response})
        # st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.write(response_text)

# List all the columns in this database

