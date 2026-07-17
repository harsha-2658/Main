"""
Simple LangGraph supervisor for CSV Q&A.

Two worker agents:
  - rag_agent  -> handles textual / descriptive questions (retrieval over row text)
  - calc_agent -> handles numeric / calculation questions (LangChain's
                  prebuilt pandas dataframe agent)

A router node (the "supervisor") looks at the question and decides which
worker should answer it. Kept intentionally simple: plain functions,
no classes, no error-handling framework, no persistence.
"""

import pandas as pd
from typing import TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State shared between nodes
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    route: str
    answer: str


# ---------------------------------------------------------------------------
# Setup helpers (called once, when a CSV is loaded)
# ---------------------------------------------------------------------------
def build_dataset_summary(df):
    """One text blob describing the dataset as a whole: columns, row count,
    and the full set of unique values for any low-cardinality (categorical)
    column. This is what lets "what categories exist" style questions get
    answered correctly, since per-row retrieval alone only sees a handful
    of rows and can't see the full set of distinct values.
    """
    lines = [
        f"This dataset has {len(df)} rows and columns: {', '.join(df.columns)}."
    ]
    for col in df.columns:
        is_text = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        if is_text and df[col].nunique() <= 50:
            values = ", ".join(sorted(df[col].astype(str).unique()))
            lines.append(f"The '{col}' column contains these unique values: {values}.")
    return " ".join(lines)


def find_exact_matches(df, question, max_rows=15):
    """Look for exact cell values (e.g. a date like '2022-01-18', or a
    category name) mentioned literally in the question, and return the
    matching rows. Semantic vector search alone often misses exact lookups
    like "the transaction on 2022-01-18" because embedding similarity
    doesn't guarantee an exact string match wins - this catches those.
    """
    matched = []
    for col in df.columns:
        for val in df[col].astype(str).unique():
            if len(val) >= 4 and val in question:
                matched.append(df[df[col].astype(str) == val])
    if not matched:
        return None
    result = pd.concat(matched).drop_duplicates()
    return result.head(max_rows)


def build_rag_chain(df, llm, embeddings):
    """Build a small FAISS index: one dataset-summary document plus one
    document per row. Returns a bundle the rag node uses at query time.
    """
    docs = [Document(page_content=build_dataset_summary(df), metadata={"type": "summary"})]
    docs += [
        Document(page_content=row.to_string(), metadata={"type": "row", "row": i})
        for i, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return {"df": df, "vectorstore": vectorstore, "llm": llm}


def build_calc_agent(df, llm):
    """LangChain's prebuilt pandas dataframe agent handles math/aggregation."""
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,  # required by langchain_experimental to run
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def make_router_node(llm):
    def router_node(state: AgentState) -> dict:
        question = state["question"]
        prompt = (
            "You are a routing assistant for a CSV question-answering system.\n"
            "Classify the user's question into exactly one category:\n\n"
            "- calculation: needs math, aggregation, sums, averages, counts, "
            "sorting, filtering by numeric conditions, comparisons, statistics.\n"
            "- textual: needs understanding, description, explanation, lookup, "
            "or general meaning of the data (non-numeric reasoning).\n\n"
            f"Question: {question}\n\n"
            "Answer with exactly one word: calculation or textual."
        )
        result = llm.invoke(prompt).content.strip().lower()
        route = "calculation" if "calc" in result else "textual"
        return {"route": route}

    return router_node


def make_rag_node(rag_bundle):
    def rag_node(state: AgentState) -> dict:
        question = state["question"]
        df = rag_bundle["df"]
        vectorstore = rag_bundle["vectorstore"]
        llm = rag_bundle["llm"]

        retrieved_docs = vectorstore.similarity_search(question, k=6)
        context_parts = [d.page_content for d in retrieved_docs]

        exact_matches = find_exact_matches(df, question)
        if exact_matches is not None and not exact_matches.empty:
            context_parts.append(
                "Exact matching rows found for this question:\n"
                + exact_matches.to_string(index=False)
            )

        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the question using only the context below, which comes "
            "from a CSV dataset. If the answer isn't in the context, say you "
            "don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        answer = llm.invoke(prompt).content
        return {"answer": answer}

    return rag_node


def make_calc_node(calc_agent):
    def calc_node(state: AgentState) -> dict:
        result = calc_agent.invoke({"input": state["question"]})
        answer = result["output"] if isinstance(result, dict) else str(result)
        return {"answer": answer}

    return calc_node


# ---------------------------------------------------------------------------
# Build the compiled graph
# ---------------------------------------------------------------------------
def build_graph(rag_bundle, calc_agent, llm):
    graph = StateGraph(AgentState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("rag_agent", make_rag_node(rag_bundle))
    graph.add_node("calc_agent", make_calc_node(calc_agent))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "textual": "rag_agent",
            "calculation": "calc_agent",
        },
    )
    graph.add_edge("rag_agent", END)
    graph.add_edge("calc_agent", END)

    return graph.compile()


# """
# Simple LangGraph supervisor for CSV Q&A.

# Two worker agents:
#   - rag_agent  -> handles textual / descriptive questions (retrieval over row text)
#   - calc_agent -> handles numeric / calculation questions (LangChain's
#                   prebuilt pandas dataframe agent)

# A router node (the "supervisor") looks at the question and decides which
# worker should answer it. Kept intentionally simple: plain functions,
# no classes, no error-handling framework, no persistence.
# """

# from typing import TypedDict
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langgraph.graph import StateGraph, END


# # ---------------------------------------------------------------------------
# # State shared between nodes
# # ---------------------------------------------------------------------------
# class AgentState(TypedDict):
#     question: str
#     route: str
#     answer: str


# # ---------------------------------------------------------------------------
# # Setup helpers (called once, when a CSV is loaded)
# # ---------------------------------------------------------------------------
# def build_rag_chain(df, llm, embeddings):
#     """Turn each row into a text Document and build a retrieval chain.

#     Uses the current LCEL-based pattern (create_retrieval_chain +
#     create_stuff_documents_chain) since the old `RetrievalQA` chain class
#     is deprecated (removed as of langchain 0.3.0).
#     """
#     docs = [
#         Document(page_content=row.to_string(), metadata={"row": i})
#         for i, row in df.iterrows()
#     ]
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are answering questions about rows from a CSV file. "
#                 "Use only the given context to answer. "
#                 "If you don't know the answer, say you don't know.\n\n"
#                 "Context:\n{context}",
#             ),
#             ("human", "{input}"),
#         ]
#     )

#     combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#     qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
#     return qa_chain


# def build_calc_agent(df, llm):
#     """LangChain's prebuilt pandas dataframe agent handles math/aggregation."""
#     return create_pandas_dataframe_agent(
#         llm,
#         df,
#         verbose=False,
#         allow_dangerous_code=True,  # required by langchain_experimental to run
#     )


# # ---------------------------------------------------------------------------
# # Graph nodes
# # ---------------------------------------------------------------------------
# def make_router_node(llm):
#     def router_node(state: AgentState) -> dict:
#         question = state["question"]
#         prompt = (
#             "You are a routing assistant for a CSV question-answering system.\n"
#             "Classify the user's question into exactly one category:\n\n"
#             "- calculation: needs math, aggregation, sums, averages, counts, "
#             "sorting, filtering by numeric conditions, comparisons, statistics.\n"
#             "- textual: needs understanding, description, explanation, lookup, "
#             "or general meaning of the data (non-numeric reasoning).\n\n"
#             f"Question: {question}\n\n"
#             "Answer with exactly one word: calculation or textual."
#         )
#         result = llm.invoke(prompt).content.strip().lower()
#         route = "calculation" if "calc" in result else "textual"
#         return {"route": route}

#     return router_node


# def make_rag_node(qa_chain):
#     def rag_node(state: AgentState) -> dict:
#         answer = qa_chain.invoke({"input": state["question"]})["answer"]
#         return {"answer": answer}

#     return rag_node


# def make_calc_node(calc_agent):
#     def calc_node(state: AgentState) -> dict:
#         result = calc_agent.invoke({"input": state["question"]})
#         answer = result["output"] if isinstance(result, dict) else str(result)
#         return {"answer": answer}

#     return calc_node


# # ---------------------------------------------------------------------------
# # Build the compiled graph
# # ---------------------------------------------------------------------------
# def build_graph(qa_chain, calc_agent, llm):
#     graph = StateGraph(AgentState)

#     graph.add_node("router", make_router_node(llm))
#     graph.add_node("rag_agent", make_rag_node(qa_chain))
#     graph.add_node("calc_agent", make_calc_node(calc_agent))

#     graph.set_entry_point("router")
#     graph.add_conditional_edges(
#         "router",
#         lambda state: state["route"],
#         {
#             "textual": "rag_agent",
#             "calculation": "calc_agent",
#         },
#     )
#     graph.add_edge("rag_agent", END)
#     graph.add_edge("calc_agent", END)

#     return graph.compile()