"""Service for managing vector store and retrieval."""
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


class VectorStoreService:
    """Responsible for vector store initialization and retrieval."""

    def __init__(self,
                 db_path: str,
                 collection_name: str,
                 embeddings_model: str,
                 retrieval_k: int = 5):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.retrieval_k = retrieval_k
        self.embeddings = OllamaEmbeddings(model=embeddings_model)
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Create vector store directory if it doesn't exist."""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        """Add documents to vector store."""
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        vector_store.add_documents(documents=documents, ids=ids)

    def get_retriever(self):
        """Get configured retriever from vector store."""
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        return vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
