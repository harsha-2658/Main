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

import re
import warnings
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
    history: str
    standalone_question: str
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


_DATE_PATTERN = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b")


def _detect_date_columns(df):
    """Columns where most values successfully parse as dates."""
    date_cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                date_cols.append(col)
    return date_cols


def find_exact_matches(df, question, max_rows=15):
    """Look for exact cell values mentioned in the question and return the
    matching rows, so lookup-style questions don't depend on embedding
    similarity alone.

    Dates get special handling: rather than requiring the question's date
    string to match the CSV's date string character-for-character (e.g.
    "05-06-2022" vs "2022-06-05"), we parse both as real dates and compare
    values. Day-first and month-first are both tried since a written date
    like "05-06-2022" is genuinely ambiguous without a locale.
    """
    matched = []
    date_cols = _detect_date_columns(df)

    for col in date_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_col = pd.to_datetime(df[col], errors="coerce")
        for candidate in _DATE_PATTERN.findall(question):
            for dayfirst in (False, True):
                parsed_candidate = pd.to_datetime(
                    candidate, errors="coerce", dayfirst=dayfirst
                )
                if pd.notna(parsed_candidate):
                    hits = df[parsed_col == parsed_candidate]
                    if not hits.empty:
                        matched.append(hits)

    for col in df.columns:
        if col in date_cols:
            continue  # already handled above
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
def make_rewrite_node(llm):
    """Resolve follow-up questions (e.g. "what about Pub?") into standalone
    questions using recent conversation history, before routing/answering.
    If there's no history, or the question is already standalone, the
    question passes through unchanged.
    """

    def rewrite_node(state: AgentState) -> dict:
        question = state["question"]
        history = state.get("history", "")
        if not history.strip():
            return {"standalone_question": question}

        prompt = (
            "You are given a conversation history and a follow-up question "
            "about a CSV dataset. Rewrite the follow-up question into a "
            "standalone question that includes any context (categories, "
            "dates, filters, etc.) implied by the history, so it can be "
            "understood on its own. If it's already standalone, return it "
            "unchanged. Reply with only the rewritten question, nothing else.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Follow-up question: {question}\n"
            "Standalone question:"
        )
        standalone = llm.invoke(prompt).content.strip()
        return {"standalone_question": standalone}

    return rewrite_node


def make_router_node(llm):
    def router_node(state: AgentState) -> dict:
        question = state["standalone_question"]
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
        question = state["standalone_question"]
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
        result = calc_agent.invoke({"input": state["standalone_question"]})
        answer = result["output"] if isinstance(result, dict) else str(result)
        return {"answer": answer}

    return calc_node


# ---------------------------------------------------------------------------
# Build the compiled graph
# ---------------------------------------------------------------------------
def build_graph(rag_bundle, calc_agent, llm):
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", make_rewrite_node(llm))
    graph.add_node("router", make_router_node(llm))
    graph.add_node("rag_agent", make_rag_node(rag_bundle))
    graph.add_node("calc_agent", make_calc_node(calc_agent))

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "router")
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