"""Web search retriever using SearchService."""
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from search_service import SearchService

logger = logging.getLogger(__name__)


class SearchRetriever(BaseRetriever):
    """Retriever that fetches documents from web search results."""

    search_service: SearchService
    num_results: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Fetch documents from web search."""
        try:
            logger.info(f"🔍 Searching web for: {query}")
            search_results = self.search_service.search(query, self.num_results)
            logger.info(f"✓ Found {len(search_results)} web results")

            documents = []
            for i, result in enumerate(search_results, 1):
                logger.info(f"  [{i}] Processing: {result.get('title', 'Untitled')[:50]}...")
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("link"),
                        "title": result.get("title"),
                    }
                )
                documents.append(doc)

            logger.info(f"✓ Converted {len(documents)} results to documents")
            return documents

        except ValueError as e:
            logger.warning(f"⚠ Web search error: {e}")
            return []
        except Exception as e:
            logger.error(f"⚠ Error fetching search results: {e}")
            return []
