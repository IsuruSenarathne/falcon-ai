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
        """Get the prompt template for LLM with improved structure."""
        return """Use the following context to answer the question.

Context:
{context}

Question:
{question}

=== MANDATORY RESPONSE FORMAT ===
Return ONLY valid JSON, nothing else. No markdown, no code blocks, no explanations.

Format:
{{"answer": "...", "reasoning": "..."}}

=== ANSWER REQUIREMENTS ===
- Include ALL specific details, names, numbers, examples
- For lists/multiple items: list them with numbers or bullet points within the answer
- Be comprehensive and detailed in the answer field
- Never put important details only in reasoning

=== REASONING REQUIREMENTS ===
- Show step-by-step logic numbered as: 1. ... 2. ... 3. ...
- Explain WHY the answer is correct
- Keep reasoning concise but complete
- Only include reasoning if question asks for explanation

=== STRICT RULES ===
1. ONLY output valid JSON - no other text
2. Never use markdown backticks or code blocks
3. Both fields must be non-empty strings
4. Answer must contain all specific information (no vague intro)
5. Do not put details in reasoning that belong in answer
6. Double-check JSON is valid before responding

Start your response with {{ and end with }}"""

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
            # Determine retriever based on context_type
            if req.context_type == "web_search" or "websearch" in req.question.lower():
                logger.info(f"Using WEB SEARCH retriever | context_type={req.context_type}")
                retriever_type = "web_search"
                retriever = SearchRetriever(search_service=self.search_service)
            elif req.context_type == "datasource":
                logger.info(f"Using DATASOURCE retriever | context_type={req.context_type}")
                retriever_type = "datasource"
                retriever = self.vector_store.get_retriever()
            else:
                # Default: user should use their own knowledge - no external context
                logger.info(f"No context_type specified | context_type={req.context_type}")
                retriever_type = "none"
                retriever = None

            # Retrieve context
            logger.debug("Step 1: Retrieving context...")
            if retriever:
                context_docs = retriever.invoke(req.question)
                context = self._format_context(context_docs)
                logger.info(f"Context retrieved | characters={len(str(context))}")

                # If context is minimal, use default message
                if not context or len(context.strip()) < 10:
                    logger.warning("Minimal context retrieved - using default knowledge prompt")
                    context = "You need to use your own knowledge to answer this question. No external context is available."
            else:
                # No retriever specified - use default message
                logger.info("Using default mode - no retriever selected")
                context = "You need to use your own knowledge to answer this question. No external context is available."

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

            # Validate response format (Issue 1 check)
            self._validate_response_format(response)

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

    def _validate_response_format(self, response: str) -> bool:
        """Validate response format (Issue 1 detection and logging).

        Returns True if format is valid, logs warnings if not.
        """
        import json

        cleaned = response.strip()

        # Check for markdown code blocks (bad format)
        if "```" in cleaned and "{" in cleaned:
            logger.warning("Response contains markdown code blocks with JSON - format may be malformed")

        # Check if response is valid JSON
        try:
            # Remove markdown if present
            if cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1]
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:]
                cleaned = cleaned.split('```')[0].strip()

            json.loads(cleaned)
            logger.debug("Response format validation: JSON is valid ✓")
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"Response format validation failed: Invalid JSON at position {e.pos}")
            return False

    @log_execution_time(logger)
    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into answer and reasoning with validation."""
        import json

        try:
            # Remove markdown code blocks if present (Issue 1 fix)
            cleaned = response.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1]
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:]
                cleaned = cleaned.split('```')[0].strip()

            # Parse JSON
            parsed = json.loads(cleaned)
            answer = parsed.get("answer", response).strip()
            reasoning = parsed.get("reasoning", "").strip()

            # Validate answer is not too vague (Issue 3 fix)
            answer = self._fix_vague_answer(answer, reasoning)

            # Ensure reasoning has proper structure (Issue 2 fix)
            reasoning = self._format_reasoning(reasoning)

            logger.debug(f"Response parsed | answer_len={len(answer)}, reasoning_len={len(reasoning)}")
            return answer, reasoning
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            # Fallback: return full response as answer
            logger.debug(f"Failed to parse JSON: {str(e)}, returning full response as answer")
            return response.strip(), ""

    def _fix_vague_answer(self, answer: str, reasoning: str) -> str:
        """Fix Issue 3: Move details from reasoning block to answer if answer is too vague."""
        # If answer is just intro/preamble (starts with generic words), move reasoning details to answer
        vague_starters = [
            "for ", "several ", "many ", "some ", "various ",
            "this ", "in ", "on ", "at ", "by ", "with "
        ]

        is_vague = (
            answer and
            answer[0].islower() and
            any(answer.lower().startswith(starter) for starter in vague_starters) and
            len(answer) < 150  # Short generic intro
        )

        if is_vague and reasoning and len(reasoning) > 200:
            logger.debug("Answer too vague, promoting details from reasoning to answer")
            # Move reasoning to answer if answer is incomplete
            return reasoning.strip() if reasoning else answer

        return answer

    def _format_reasoning(self, reasoning: str) -> str:
        """Ensure reasoning has proper numbered steps (Issue 2 fix)."""
        if not reasoning:
            return reasoning

        # Check if reasoning already has numbered steps
        if any(f"{i}." in reasoning for i in range(1, 5)):
            return reasoning

        # If reasoning exists but has no structure, try to add numbering
        sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
        if len(sentences) > 1:
            numbered = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(sentences[:5])])
            logger.debug(f"Restructured reasoning with numbered steps")
            return numbered

        return reasoning

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
            answer=answer,
            reasoning=reasoning if reasoning else None,
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
