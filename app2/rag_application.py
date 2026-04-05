"""Main RAG application orchestrator."""
import logging
from document_loader import DocumentLoader
from vector_store_service import VectorStoreService
from llm_chain_service import LLMChainService
from response_formatter import ResponseFormatter
from search_service import SearchService
from search_retriever import SearchRetriever
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGApplication:
    """Orchestrates RAG pipeline: document loading, retrieval, and LLM inference."""

    def __init__(self):
        logger.info("Initializing RAG Application...")
        # Initialize services
        logger.info("Loading documents from CSV...")
        self.document_loader = DocumentLoader(config.CSV_FILE_PATH)

        logger.info("Initializing vector store...")
        self.vector_store = VectorStoreService(
            db_path=config.VECTOR_DB_PATH,
            collection_name=config.COLLECTION_NAME,
            embeddings_model=config.EMBEDDINGS_MODEL,
            retrieval_k=config.RETRIEVAL_K
        )

        logger.info("Initializing search service...")
        self.search_service = SearchService()

        logger.info("Initializing LLM chain...")
        self.llm_chain = LLMChainService(
            model=config.LLM_MODEL,
            prompt_template=config.SYSTEM_PROMPT,
            temperature=config.LLM_TEMPERATURE
        )
        self.formatter = ResponseFormatter()
        self._initialize_vector_store()
        logger.info("RAG Application initialized successfully")

    def _initialize_vector_store(self) -> None:
        """Load documents and populate vector store on first run."""
        logger.info("Loading documents into vector store...")
        docs, ids = self.document_loader.load()
        logger.info(f"Loaded {len(docs)} documents")
        self.vector_store.add_documents(docs, ids)
        logger.info("Documents added to vector store")

    def query(self, question: str) -> str:
        """Execute a single RAG query."""
        logger.info("=" * 60)
        logger.info(f"New query: {question[:60]}...")

        # Use web search if "websearch" keyword is in the question
        if "websearch" in question.lower():
            logger.info("Using WEB SEARCH retriever")
            retriever = SearchRetriever(search_service=self.search_service)
        else:
            logger.info("Using VECTOR STORE retriever")
            retriever = self.vector_store.get_retriever()

        logger.info("Step 1: Retrieving context...")
        context = retriever.invoke(question)
        logger.info(f"Retrieved {len(str(context))} characters of context")

        logger.info("Step 2: Generating response with LLM...")
        response = self.llm_chain.invoke(question, context)
        logger.info("Response generated")
        logger.info("=" * 60)

        return response

    def run_interactive(self) -> None:
        """Run interactive query loop."""
        while True:
            question = input("\nEnter your question (or 'exit' to quit): ").strip()
            if question.lower() == "exit":
                break

            response = self.query(question)
            self.formatter.display(response)
