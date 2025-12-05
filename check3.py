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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage,RemoveMessage
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

llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------------------------
# Assistant node: invokes the LLM with tools bound
# --------------------------------------------------------------------
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def human_feedback(state:MessagesState):
    pass

def assistant(state:MessagesState):
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

builder=StateGraph(MessagesState)

builder.add_node("assistant",assistant)
builder.add_node("tools",ToolNode(tools))
builder.add_node("human_feedback",human_feedback)

builder.add_edge(START,"human_feedback")
builder.add_edge("human_feedback","assistant")
builder.add_conditional_edges(
"assistant",
tools_condition,)
builder.add_edge("tools","human_feedback")
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_feedback"],
)

thread = {"configurable": {"thread_id": "2"}}
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()

user_input=input("Tell me how you want to update state:")
# graph.update_state(thread,{"messages":user_input},as_node="human_feedback")

current=graph.get_state(thread).values.get("messages",[])

removals=[RemoveMessage(id=m.id) for m in current if isinstance(m,HumanMessage)]

graph.update_state(
    thread,
    {"messages":removals+[HumanMessage(content=user_input)]},
    as_node="human_feedback",
)

resume_cmd = Command(resume="continue")

print("\n=== Resume execution (assistant will now run) ===")
for event in graph.stream(None, thread, stream_mode="values", command=resume_cmd):
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()
