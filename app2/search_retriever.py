"""Web search retriever using SearchService."""
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from search_service import SearchService


class SearchRetriever(BaseRetriever):
    """Retriever that fetches documents from web search results."""

    search_service: SearchService
    num_results: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Fetch documents from web search."""
        try:
            search_results = self.search_service.search(query, self.num_results)
            documents = []

            for result in search_results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("link"),
                        "title": result.get("title"),
                    }
                )
                documents.append(doc)

            return documents

        except ValueError as e:
            print(f"⚠ Web search error: {e}")
            return []
        except Exception as e:
            print(f"⚠ Error fetching search results: {e}")
            return []
