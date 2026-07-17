"""
Streamlit frontend for the CSV Supervisor Multi-Agent app.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from graph1 import build_rag_chain, build_calc_agent, build_graph
from dotenv import load_dotenv
load_dotenv()

def format_history(chat_history, max_turns=4):
    """Turn the last few (role, text) chat entries into a plain text block
    the rewrite node can use to resolve follow-up questions."""
    recent = chat_history[-(max_turns * 2):]
    lines = [f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in recent]
    return "\n".join(lines)


st.set_page_config(page_title="CSV Supervisor Agent", page_icon="📊")
st.title("📊 CSV Supervisor Multi-Agent QA")
st.caption(
    "One supervisor routes your question to a RAG agent (textual questions) "
    "or a pandas calculation agent (numeric questions)."
)

# ---------------------------------------------------------------------------
# Sidebar: API key + file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    # api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("---")
    st.markdown(
        "**Examples**\n\n"
        "- Textual: *\"What does the `status` column represent?\"*\n"
        "- Calculation: *\"What is the average of the `price` column?\"*"
    )

# if not api_key:
#     st.info("Enter your OpenAI API key in the sidebar to get started.")
#     st.stop()

# os.environ["OPENAI_API_KEY"] = api_key

if not uploaded_file:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Load CSV once and build the graph once (cached in session_state)
# ---------------------------------------------------------------------------
df = pd.read_csv(uploaded_file)
st.subheader("Preview")
st.dataframe(df.head())

file_id = uploaded_file.name + str(uploaded_file.size)
if st.session_state.get("file_id") != file_id:
    with st.spinner("Setting up agents..."):
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # embeddings = OpenAIEmbeddings()
        llm_1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GOOGLE_API_KEY_6"), temperature=0)
        llm_2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",api_key=os.getenv("GOOGLE_API_KEY_6"), temperature=0)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        qa_chain = build_rag_chain(df, llm_1, embeddings)
        calc_agent = build_calc_agent(df, llm_1)

        st.session_state.graph = build_graph(qa_chain, calc_agent, llm_2)
        st.session_state.file_id = file_id
        st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
st.subheader("Ask a question")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

question = st.chat_input("Ask about the data (textual or calculation)...")

if question:
    history_text = format_history(st.session_state.chat_history)

    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke(
                {
                    "question": question,
                    "history": history_text,
                    "standalone_question": "",
                    "route": "",
                    "answer": "",
                }
            )
            answer = result["answer"]
            route = result["route"]
            standalone = result["standalone_question"]
        st.write(answer)
        caption = f"routed to: {route} agent"
        if standalone != question:
            caption += f"  |  resolved as: \"{standalone}\""
        st.caption(caption)

    st.session_state.chat_history.append(("assistant", answer))