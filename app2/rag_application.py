"""Main RAG application orchestrator."""
from document_loader import DocumentLoader
from vector_store_service import VectorStoreService
from llm_chain_service import LLMChainService
from response_formatter import ResponseFormatter
from search_service import SearchService
from search_retriever import SearchRetriever
import config


class RAGApplication:
    """Orchestrates RAG pipeline: document loading, retrieval, and LLM inference."""

    def __init__(self):
        # Initialize services
        self.document_loader = DocumentLoader(config.CSV_FILE_PATH)
        self.vector_store = VectorStoreService(
            db_path=config.VECTOR_DB_PATH,
            collection_name=config.COLLECTION_NAME,
            embeddings_model=config.EMBEDDINGS_MODEL,
            retrieval_k=config.RETRIEVAL_K
        )

        # Initialize search service
        self.search_service = SearchService()

        self.llm_chain = LLMChainService(
            model=config.LLM_MODEL,
            prompt_template=config.SYSTEM_PROMPT,
            temperature=config.LLM_TEMPERATURE
        )
        self.formatter = ResponseFormatter()
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Load documents and populate vector store on first run."""
        docs, ids = self.document_loader.load()
        self.vector_store.add_documents(docs, ids)

    def query(self, question: str) -> str:
        """Execute a single RAG query."""
        # Use web search if "websearch" keyword is in the question
        if "websearch" in question.lower():
            retriever = SearchRetriever(search_service=self.search_service)
        else:
            retriever = self.vector_store.get_retriever()

        context = retriever.invoke(question)
        response = self.llm_chain.invoke(question, context)
        return response

    def run_interactive(self) -> None:
        """Run interactive query loop."""
        while True:
            question = input("\nEnter your question (or 'exit' to quit): ").strip()
            if question.lower() == "exit":
                break

            response = self.query(question)
            self.formatter.display(response)
