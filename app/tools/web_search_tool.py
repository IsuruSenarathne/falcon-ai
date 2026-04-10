"""Web search tool definition."""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

_search_service = None


def init(search_service):
    global _search_service
    _search_service = search_service


class WebSearchInput(BaseModel):
    """Input for web search queries."""
    query: str = Field(description="The search query to look up on the web")
    num_results: int = Field(default=5, description="Number of results to fetch")


@tool(args_schema=WebSearchInput)
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for current, real-time information about the given query.
    Use this tool when the question requires up-to-date or external information.
    """
    logger.info(f"web_search tool called | query={query[:60]}")
    results = _search_service.search(query, num_results=num_results)
    if not results:
        return "No web search results found."
    parts = []
    for r in results:
        parts.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('link', '')}\n"
            f"Content: {r.get('content', '')[:800]}"
        )
    return "\n\n---\n\n".join(parts)
