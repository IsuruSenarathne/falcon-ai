"""Knowledge base search tool definition."""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

_vector_store = None


def init(vector_store):
    global _vector_store
    _vector_store = vector_store


class KnowledgeBaseInput(BaseModel):
    """Input for university knowledge base queries."""
    query: str = Field(description="The search query to look up in the knowledge base")
    num_results: int = Field(default=5, description="Number of documents to retrieve")


@tool(args_schema=KnowledgeBaseInput)
def knowledge_base_search(query: str, num_results: int = 5) -> str:
    """Search the university knowledge base for information about courses,
    advisors, departments, and modules.
    Use this tool when the question is about university-specific information.
    """
    logger.info(f"knowledge_base_search tool called | query={query[:60]}")
    retriever = _vector_store.get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(
        doc.page_content for doc in docs if doc.page_content
    )
