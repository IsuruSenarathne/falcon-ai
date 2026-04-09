"""LangChain agent middleware utilities."""
import os
from typing import Any
from langchain.messages import RemoveMessage #type: ignore
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState
from langchain.agents.middleware import before_model #type: ignore
from langgraph.runtime import Runtime

MAX_MESSAGES = int(os.getenv("AGENT_MAX_MESSAGES", "6"))


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the first message plus the most recent MAX_MESSAGES to fit context window."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES:
        return None

    first_msg = messages[0]
    recent = messages[-MAX_MESSAGES:]
    new_messages = [first_msg] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages,
        ]
    }
