# interrupt_before_assistant_run_groq_fixed.py
from dotenv import load_dotenv
import os

# Groq LLM
from langchain_groq import ChatGroq

# LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.types import Command

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool  # <-- decorator to convert functions into BaseTool

# --------------------------------------------------------------------
# Environment & LLM setup
# --------------------------------------------------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key  # ensure set for ChatGroq

# You can swap to a Groq model known to support tool calling, e.g. "llama-3.1-8b-instant"
# See: https://docs.langchain.com/oss/python/integrations/chat/groq
llm = ChatGroq(model="openai/gpt-oss-20b")

# --------------------------------------------------------------------
# Arithmetic tools (decorate so they become BaseTool and serialize correctly)
# --------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]

# Bind tools to the LLM using the recommended API
# This ensures the tool schema is converted properly for the provider.
# Docs: https://reference.langchain.com/python/langchain/tools/
llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------------------------
# Assistant node: invokes the LLM with tools bound
# --------------------------------------------------------------------
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

def assistant(state: MessagesState):
    # state["messages"] is a list of BaseMessage objects
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# --------------------------------------------------------------------
# Build graph with ToolNode
# --------------------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))  # ToolNode expects BaseTool instances

builder.add_edge(START, "assistant")
# If the AIMessage has tool_calls, route to tools; otherwise go to END
builder.add_conditional_edges("assistant", tools_condition, ["tools", END])
builder.add_edge("tools", "assistant")

# --------------------------------------------------------------------
# Compile with a static breakpoint BEFORE the assistant
# --------------------------------------------------------------------
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["assistant"],  # <-- pause BEFORE assistant runs
)

# --------------------------------------------------------------------
# Demo: run until interrupt, then resume (no approval logic, just continue)
# --------------------------------------------------------------------
thread = {"configurable": {"thread_id": "1"}}
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

print("=== Run until interrupt (before assistant) ===")
for event in graph.stream(initial_input, thread, stream_mode="values"):
    # At the interrupt, you’ll get the current values (messages so far)
    print(event)

# Optional: inspect state at the breakpoint
state_snapshot = graph.get_state(thread)
print("\n--- State at breakpoint ---")
print(state_snapshot.values)

# Resume execution; no approval—just a token to continue
resume_cmd = Command(resume="continue")

print("\n=== Resume execution (assistant will now run) ===")
for event in graph.stream(None, thread, stream_mode="values", command=resume_cmd):
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()
