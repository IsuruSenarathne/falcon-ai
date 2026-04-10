"""LangChain agent middleware utilities."""
import os
from typing import Any
from langchain.messages import RemoveMessage, ToolMessage #type: ignore
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState
from langchain.agents.middleware import before_model, wrap_tool_call #type: ignore
from langgraph.runtime import Runtime
from app.utils.logger import get_logger

logger = get_logger(__name__)

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


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Return a structured error ToolMessage instead of raising on tool failure."""
    try:
        return handler(request)
    except Exception as e:
        logger.warning(f"Tool '{request.tool_call.get('name', 'unknown')}' failed: {e}")
        return ToolMessage(
            content=f"<p>Tool error: {str(e)}. Please check your input and try again.</p>",
            tool_call_id=request.tool_call["id"],
        )
