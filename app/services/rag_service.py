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
        return """Context:
{context}

Question: {question}

Respond using ONLY this exact structure (no other text before or after):

ANSWER: [inner HTML using <p>, <ul><li>, <ol><li>, or <table><thead><tr><th/></tr></thead><tbody><tr><td/></tr></tbody></table> - every table row must have <tr>]

REASONING: [numbered plain text steps: 1. ... 2. ... 3. ...]"""

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
        """Validate response format follows ANSWER: (HTML) ... REASONING: (plain) ... structure.

        Returns True if format is valid, logs warnings if not.
        """
        cleaned = response.strip()

        # Check for JSON objects (bad format) - but not HTML
        import re
        has_json = bool(re.search(r'\{\s*"[^"]+"\s*:', cleaned))
        if has_json:
            logger.warning("Response contains JSON formatting - should be HTML answer with plain text reasoning")
            return False

        if "```" in cleaned:
            logger.warning("Response contains markdown code blocks - answer should be HTML")
            return False

        # Ensure response is not empty
        if len(cleaned) < 5:
            logger.warning("Response is too short or empty")
            return False

        if not re.search(r'(?i)ANSWER\s*:', cleaned):
            logger.warning("Response missing ANSWER: section header")

        logger.debug("Response format validation passed")
        return True

    @log_execution_time(logger)
    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response. Expects ANSWER: ... REASONING: ... but handles model deviations."""
        import re

        try:
            cleaned = response.strip()
            cleaned = self._remove_json_blocks(cleaned)

            answer = ""
            reasoning = ""

            # Case-insensitive match for ANSWER: and REASONING: markers
            answer_match = re.search(r'(?i)ANSWER\s*:', cleaned)
            reasoning_match = re.search(r'(?i)REASONING\s*:', cleaned)

            if answer_match and reasoning_match:
                answer = cleaned[answer_match.end():reasoning_match.start()].strip()
                reasoning = cleaned[reasoning_match.end():].strip()
            elif answer_match:
                answer = cleaned[answer_match.end():].strip()
                reasoning = ""
            else:
                # Model ignored markers — extract HTML as answer, numbered list as reasoning
                logger.warning("Model did not follow ANSWER:/REASONING: format, attempting fallback extraction")
                answer, reasoning = self._fallback_extract(cleaned)

            # Repair common HTML issues (e.g. missing <tr> in tables)
            answer = self._repair_html(answer)

            # Remove duplicates
            answer = self._remove_duplicates(answer)

            if reasoning:
                reasoning = self._format_reasoning(reasoning)

            logger.debug(f"Response parsed | answer_len={len(answer)}, reasoning_len={len(reasoning)}")
            return answer, reasoning
        except Exception as e:
            logger.debug(f"Failed to parse response: {str(e)}, returning full response as answer")
            return response.strip(), ""

    def _fallback_extract(self, text: str) -> tuple:
        """Fallback: extract HTML blocks as answer and numbered steps as reasoning."""
        import re

        # Extract all HTML content
        html_blocks = re.findall(r'<[^>]+>.*?</[^>]+>', text, re.DOTALL)
        if html_blocks:
            answer = "".join(html_blocks)
        else:
            # No HTML - strip prose preamble (sentences before any useful content)
            answer = text

        # Extract numbered steps as reasoning (lines starting with digit.)
        numbered_lines = re.findall(r'^\s*\d+\.\s+.+', text, re.MULTILINE)
        reasoning = "\n".join(numbered_lines) if numbered_lines else ""

        # If no numbered steps found, build a minimal reasoning
        if not reasoning:
            reasoning = "1. Retrieved relevant information from context.\n2. Formatted the answer based on the question."

        return answer, reasoning

    def _repair_html(self, html: str) -> str:
        """Fix common LLM HTML mistakes, especially missing <tr> wrappers in tables."""
        import re

        def fix_tbody(match):
            tbody_content = match.group(1)
            # If <td> exists without a wrapping <tr>, group them into rows
            if '<td>' in tbody_content and '<tr>' not in tbody_content:
                tds = re.findall(r'<td>.*?</td>', tbody_content, re.DOTALL)
                # Detect column count from <thead>
                th_count = len(re.findall(r'<th>', html))
                cols = th_count if th_count > 0 else 2

                rows = []
                for i in range(0, len(tds), cols):
                    row_cells = tds[i:i + cols]
                    rows.append(f"<tr>{''.join(row_cells)}</tr>")
                return f"<tbody>{''.join(rows)}</tbody>"
            return match.group(0)

        # Fix tbody missing <tr> wrappers
        html = re.sub(r'<tbody>(.*?)</tbody>', fix_tbody, html, flags=re.DOTALL)
        return html

    def _remove_json_blocks(self, text: str) -> str:
        """Remove JSON code blocks from response if LLM included them."""
        cleaned = text.strip()

        # Remove markdown code blocks
        if cleaned.startswith('```'):
            try:
                parts = cleaned.split('```')
                if len(parts) >= 3:
                    # Extract content between backticks
                    content = parts[1]
                    if content.startswith('json'):
                        content = content[4:]
                    cleaned = content.strip()
            except Exception as e:
                logger.debug(f"Failed to remove code blocks: {e}")

        return cleaned

    def _remove_duplicates(self, text: str) -> str:
        """Remove duplicate sentences/blocks from response."""
        lines = text.split('\n')
        seen = set()
        unique_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line)
            elif not line_stripped:
                # Keep empty lines for formatting
                unique_lines.append(line)

        return '\n'.join(unique_lines).strip()

    def _ensure_complete_answer(self, answer: str) -> str:
        """Ensure answer is complete and not just a vague intro."""
        if not answer:
            return answer

        # Check if answer starts with a vague intro
        vague_starters = [
            "for ", "several ", "many ", "some ", "various ",
            "this ", "in ", "on ", "at ", "by ", "with ",
            "it ", "there ", "you can"
        ]

        is_vague_intro = (
            answer[0].islower() and
            any(answer.lower().startswith(starter) for starter in vague_starters) and
            len(answer) < 150
        )

        if is_vague_intro:
            logger.debug(f"Answer detected as vague intro only, flagging for improvement")
            # Return as-is, but log it - the LLM should have provided better answer
            return answer

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
        """Process successful query. Answer is already HTML from LLM."""
        # Combine answer (HTML) + reasoning (plain text converted to HTML) for DB storage
        db_answer = self._combine_for_storage(answer, reasoning)

        logger.debug(f"Saving successful query to DB")
        conversation_id, bot_msg = ConversationService.save_exchange(
            question=req.question,
            answer=db_answer,
            status=MessageStatus.SUCCESS,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        logger.debug(f"Query saved | conversation_id={conversation_id}")
        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=answer,           # HTML answer sent in response
            reasoning=reasoning if reasoning else None,
            status="success",
            response_time=0,
            created_at=bot_msg.created_at.isoformat(),
        )

    def _combine_for_storage(self, answer: str, reasoning: str) -> str:
        """Combine HTML answer and plain text reasoning for DB storage."""
        parts = []

        if answer:
            parts.append(answer)  # already HTML

        if reasoning:
            reasoning_html = self._reasoning_to_html(reasoning)
            parts.append(f"<p><strong>Reasoning:</strong></p>{reasoning_html}")

        return "".join(parts) if parts else "<p>No response generated.</p>"

    def _reasoning_to_html(self, reasoning: str) -> str:
        """Convert plain text numbered reasoning to HTML ordered list."""
        lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
        items = []

        for line in lines:
            if line and line[0].isdigit() and '.' in line[:3]:
                items.append(f"<li>{line[line.index('.')+1:].strip()}</li>")
            else:
                items.append(f"<li>{line}</li>")

        if items:
            return f"<ol>{''.join(items)}</ol>"
        return f"<p>{reasoning}</p>"

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
