# import streamlit as st
# from pathlib import Path
# from langchain.agents import create_sql_agent
# from langchain.sql_database import SQLDatabase
# from langchain.agents.agent_types import AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from sqlalchemy import create_engine
# import sqlite3
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()  # This will load environment variables from your .env file


# st.set_page_config(page_title="Langchain: CHat with SQL DB")
# st.title("Langchain: Chat with SQL")

# INJECTION_WARNING=  """
#                     SQL agent can be vulnerable to prompt injection. Use a DB role with limited permissions.
#                     Read more [here](https://python.langchain.com/docs/security).
#                     """

# LOCALDB="USE_LOCALDB"
# MYSQL="USE_MYSQL"

# # radio
# radio_opt=["Use SQLLie 3 database-Student.db","Connect to you SQL Database"]

# selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

# if radio_opt.index(selected_opt)==1:
#     db_uri=MYSQL
#     mysql_host=st.sidebar.text_input("Provide MY SQL Host")
#     mysql_user=st.sidebar.text_input("MySQL User")
#     mysql_password=st.sidebar.text_input("MySQL password", type="password")
#     mysql_db=st.sidebar.text_input("MySQL database")
# else:
#     db_uri=LOCALDB

# api_key=st.sidebar.text_input(label="Groq api key",type="password")

# if not db_uri:
#     st.info("Please enter the database information and uri")

# if not api_key:
#     st.info("Please add the groq api key")


# llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile",streaming=True)

# @st.cache_resource(ttl="2h")
# def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
#     if db_uri==LOCALDB:
#         dbfilepath=(Path(__file__).parent/"student.db").absolute()
#         print(dbfilepath)
#         creator=lambda:sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
#         return SQLDatabase(create_engine("sqlite:///",creator=creator))
#     elif db_uri==MYSQL:
#         if not (mysql_host and mysql_user and mysql_password and mysql_db):
#             st.error("Please provide all MySQL connection details")
#             st.stop()
#         return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
# if db_uri==MYSQL:
#     db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
# else:
#     db=configure_db(db_uri)

# # toolkit 
# toolkit=SQLDatabaseToolkit(db=db,llm=llm)

# agent=create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

# if 'messages' not in st.session_state or st.sidebar.button("Clear message history"):
#     st.session_state["messages"]=[{"role":"assistant","content":"How can I help you?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write([msg["content"]])

# user_query=st.chat_input(placeholder="Ask anything form the database")

# if user_query:
#     st.session_state.messages.append({"role":"user","content":user_query})
#     st.chat_message("user").write(user_query)

#     with st.chat_message("assistant"):
#         streamlit_callback=StreamlitCallbackHandler(st.container())
#         response=agent.run(user_query,callbacks=[streamlit_callback])
#         st.session_state.messages.append({"role":"assistant","content":response})
#         st.write(response)


import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
# from langchain.sql_database import SQLDatabase
# from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain : chat with sql DB")
st.title("Langchain : Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

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

api_key = st.sidebar.text_input("Enter GROQ API Key" , type="password")

if not db_url:
    st.info("Please enter the database information and url")

if not api_key:
    st.info("Please enter API kay")

#LLM Model

llm = ChatGroq(groq_api_key = api_key , model_name = "openai/gpt-oss-20b" , streaming=True)

# @st.cache_resource(ttl="2h")
# def configure_db(db_url, mysql_host=None , mysql_user= None , mysql_password=None , mysql_db=None):
#     if db_url==LOCALDB:
#         dbfilepath=(Path(__file__).parent/"student.db").absolute()
#         print(dbfilepath)
#         creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro" , uri=True)
#         return SQLDatabase(create_engine("sqlite:///", creator=creator))
#     elif db_url==MYSQL:
#         if not (mysql_host and mysql_user and mysql_password and mysql_db):
#             st.error("Please provide all MYSQL connection details.")
#             st.stop()
#         return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    

@st.cache_resource(ttl="2h")
def configure_db(db_url, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_url == LOCALDB:
        dbfilepath = Path(__file__).parent / "student.db"

        if not dbfilepath.exists():
            st.error(f"Database file not found at {dbfilepath}")
            st.stop()

        # Use the absolute path directly
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

#Toolkit

toolkit = SQLDatabaseToolkit(db=db ,llm=llm)

# agent = create_sql_agent(
#     llm = llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

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
        response = agent.run(user_query , callbacks=[streamlit_callback])
        st.session_state.messages.append({"role" : "assistant" , "content" : response})
        st.write(response)
