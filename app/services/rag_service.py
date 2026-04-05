"""Simplified RAG service using app2 architecture."""
import time
from langchain_core.documents import Document
from app.services.vector_store_service import VectorStoreService
from app.services.llm_chain_service import LLMChainService
from app.services.search_service import SearchService
from app.services.search_retriever import SearchRetriever
from app.repositories.knowledge_repository import KnowledgeRepository
from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.services.conversation_service import ConversationService
from app.constants.models import LLM_MAIN_MODEL, EMBEDDING_MODEL
from app.utils.logger import get_logger, log_service_call, log_execution_time, log_errors, write_context_to_file

logger = get_logger(__name__)


class RAGService:
    """Simplified RAG service combining vector store and web search."""

    def __init__(self):
        logger.info("Initializing RAG Service...")

        # Load documents from database
        logger.info("Loading documents from knowledge repository...")
        documents = KnowledgeRepository.load_documents()
        docs = [Document(page_content=doc) for doc in documents]
        logger.info(f"Loaded {len(docs)} documents")

        # Initialize vector store
        logger.info("Initializing vector store...")
        self.vector_store = VectorStoreService(
            db_path="chroma_db",
            collection_name="knowledge_base",
            embeddings_model=EMBEDDING_MODEL,
            retrieval_k=5
        )
        self.vector_store.add_documents(docs, [str(i) for i in range(len(docs))])

        # Initialize search service
        logger.info("Initializing search service...")
        self.search_service = SearchService()

        # Initialize LLM chain
        logger.info("Initializing LLM chain...")
        self.llm_chain = LLMChainService(
            model=LLM_MAIN_MODEL,
            prompt_template=self._get_prompt_template(),
            temperature=0.7
        )

        logger.info("RAG Service initialized successfully")

    def _get_prompt_template(self) -> str:
        """Get the prompt template for LLM."""
        return """You are a helpful assistant. CRITICAL INSTRUCTIONS:

1. **USE ONLY THE PROVIDED CONTEXT**: Answer based EXCLUSIVELY on the context provided below.
2. **IGNORE TRAINING DATA**: Do NOT use your training knowledge if it contradicts the provided context.
3. **CONTEXT IS AUTHORITATIVE**: If the context contains information, that is the source of truth.
4. **IF NOT IN CONTEXT**: If the answer is not in the provided context, explicitly state: "This information is not available in the provided context."

Context:
{context}

Question: {question}

Provide your answer in this format:
---ANSWER---
[Your detailed answer based ONLY on the provided context]
---REASONING---
[Explain how the answer comes from the provided context, cite specific parts if relevant]
"""

    @log_service_call(logger)
    @log_errors(logger)
    def query(self, req: QueryRequest) -> QueryResponse:
        """Execute a query using vector store or web search."""
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Query started | question={req.question[:60]}... | user_id={req.user_id}")
        query_start_time = time.time()
        retriever_type = None

        try:
            # Determine retriever
            if "websearch" in req.question.lower():
                logger.info("Using WEB SEARCH retriever")
                retriever_type = "web_search"
                retriever = SearchRetriever(search_service=self.search_service)
            else:
                logger.info("Using VECTOR STORE retriever")
                retriever_type = "vector_store"
                retriever = self.vector_store.get_retriever()

            # Retrieve context
            logger.debug("Step 1: Retrieving context...")
            context_docs = retriever.invoke(req.question)
            context = self._format_context(context_docs)
            logger.info(f"Context retrieved | characters={len(str(context))}")

            # Log context availability for debugging
            if not context or len(context.strip()) < 10:
                logger.warning("Minimal context retrieved - LLM may fall back to training data")

            # Write context to latest_context.txt (overwrites previous)
            conv_id = req.session_id or req.conversation_id or "unknown"
            write_context_to_file(
                question=req.question,
                context=context,
                retriever_type=retriever_type,
                conversation_id=conv_id,
            )

            # Write context to debug folder (JSON format)
            logger.debug(f"Debug context written | conversation_id={conv_id}")

            # Generate response
            logger.debug("Step 2: Generating response with LLM...")
            response = self.llm_chain.invoke(req.question, context)
            logger.debug("LLM response generated")

            # Validate response uses context
            self._validate_context_usage(context, response, req.question)

            # Parse and save
            answer, reasoning = self._parse_response(response)
            result = self._process_success(req, answer, reasoning)

            execution_time_ms = (time.time() - query_start_time) * 1000
            logger.info(f"Query completed successfully | execution_time={execution_time_ms:.2f}ms")

            return result

        except Exception as e:
            execution_time_ms = (time.time() - query_start_time) * 1000
            logger.error(f"Query failed: {str(e)}")

            return self._process_error(req, str(e))

    def _format_context(self, docs) -> str:
        """Format documents into context string."""
        if isinstance(docs, list):
            return "\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs])
        else:
            return docs.page_content if hasattr(docs, 'page_content') else str(docs)

    def _validate_context_usage(self, context: str, response: str, question: str) -> bool:
        """Validate that response appears to use the provided context.

        Returns True if response seems context-grounded, False if it may be using training data.
        """
        # Check if response explicitly states context insufficiency
        if "not available in the provided context" in response.lower():
            return True  # Honest response about context

        # For web search queries with recent info, check if response contains date markers
        if "web" in question.lower() and ("2025" in context or "2024" in context):
            if "2025" not in response and "2024" not in response:
                logger.warning("Response may not be using current context dates")
                return False

        return True

    @log_execution_time(logger)
    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into answer and reasoning."""
        answer = response
        reasoning = ""

        # Parse with new format markers
        if "---ANSWER---" in response and "---REASONING---" in response:
            answer_start = response.find("---ANSWER---") + len("---ANSWER---")
            reasoning_start = response.find("---REASONING---") + len("---REASONING---")
            answer = response[answer_start:reasoning_start].replace("---REASONING---", "").strip()
            reasoning = response[reasoning_start:].strip()
        # Fallback to old format for backwards compatibility
        elif "---REASONING---" in response:
            parts = response.split("---REASONING---")
            answer = parts[0].strip()
            reasoning = parts[1].strip() if len(parts) > 1 else ""

        # Warn if model didn't use context properly
        if "not available in the provided context" in answer.lower():
            logger.warning("Model reported context insufficiency")

        logger.debug(f"Response parsed | answer_len={len(answer)}, reasoning_len={len(reasoning)}")
        return answer, reasoning

    @log_execution_time(logger)
    def _process_success(self, req: QueryRequest, answer: str, reasoning: str) -> QueryResponse:
        """Process successful query."""
        formatted_answer = answer
        if reasoning:
            formatted_answer += f"\n\nReasoning:\n{reasoning}"

        logger.debug(f"Saving successful query to DB")
        conversation_id, bot_msg = ConversationService.save_exchange(
            question=req.question,
            answer=formatted_answer,
            status=MessageStatus.SUCCESS,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        logger.debug(f"Query saved | conversation_id={conversation_id}")
        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=formatted_answer,
            status="success",
            response_time=0,
            created_at=bot_msg.created_at.isoformat(),
        )

    @log_execution_time(logger)
    def _process_error(self, req: QueryRequest, error: str) -> QueryResponse:
        """Process failed query."""
        logger.warning(f"Saving failed query to DB | error={error}")
        conversation_id, _ = ConversationService.save_exchange(
            question=req.question,
            answer=None,
            status=MessageStatus.ERROR,
            error=error,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        logger.debug(f"Error query saved | conversation_id={conversation_id}")
        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=None,
            status="error",
            response_time=0,
            created_at="",
            error=error,
        )
