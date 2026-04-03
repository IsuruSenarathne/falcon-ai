"""Entry point for RAG application."""
from rag_application import RAGApplication


if __name__ == "__main__":
    app = RAGApplication()
    app.run_interactive()