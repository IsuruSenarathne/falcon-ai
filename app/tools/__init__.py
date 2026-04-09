"""Tools package — exports all agent tools and a single init function."""
from app.tools import knowledge_base_tool
from app.tools.web_search_tool import web_search
from app.tools.knowledge_base_tool import knowledge_base_search
from app.tools import web_search_tool


def init_tools(search_service, vector_store):
    """Initialise all tools with their required service instances. Call once at startup."""
    web_search_tool.init(search_service)
    knowledge_base_tool.init(vector_store)


__all__ = ["init_tools", "web_search", "knowledge_base_search"]
