from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants.models import LLM_MAIN_MODEL, EMBEDDING_MODEL
from app.dto.conversation_dto import QueryRequest, QueryResponse, SearchRequest
from app.models.conversation import MessageStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.conversation_service import ConversationService
from app.services.search_service import SearchService


class RAGService:

    def __init__(self, task_breakdown_service=None):
        self.task_breakdown_service = task_breakdown_service
        self.search_service = SearchService(task_breakdown_service=task_breakdown_service)
        
        documents = KnowledgeRepository.load_documents()
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
        print(f"✓ Knowledge base loaded: {len(documents)} documents from database")

        # Template for knowledge base context
        self.kb_template = """Answer the question based ONLY on the following context:
{context}

{conversation_context}

Question: {question}

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Do NOT include brackets, labels, or placeholder text. Just the actual content.

Example format:
<p>Your answer here</p>

---REASONING---
Your reasoning here

Rules:
- Use valid HTML tags for formatting
- Explain how you derived the answer from the context
- Do NOT include diagrams, charts, or images
- Inner HTML only (no <html>, <head>, <body>, <style>, <script> tags)
"""

        # Template for general knowledge (no context)
        self.general_template = """Answer the following question using your knowledge:

{conversation_context}

Question: {question}

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Do NOT include brackets, labels, or placeholder text. Just the actual content.

Example format:
<p>Your answer here</p>

---REASONING---
Your reasoning here

Rules:
- Use valid HTML tags for formatting
- Explain your reasoning
- Do NOT include diagrams, charts, or images
- Inner HTML only (no <html>, <head>, <body>, <style>, <script> tags)
"""

        self.llm = ChatOllama(model=LLM_MAIN_MODEL)

    def _parse_response(self, raw_response: str) -> tuple[str, str]:
        """Parse raw response into answer and reasoning sections."""
        answer = ""
        reasoning = ""

        # Parse by ---REASONING--- separator
        if "---REASONING---" in raw_response:
            parts = raw_response.split("---REASONING---")
            answer = parts[0].strip()
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        elif "REASONING:" in raw_response:
            # Fallback for old ANSWER:/REASONING: format
            parts = raw_response.split("REASONING:")
            answer = parts[0].replace("ANSWER:", "").strip()
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Last resort fallback
            answer = raw_response.strip()
            reasoning = ""

        return answer, reasoning

    def _format_with_tasks_and_reasoning(self, answer: str, reasoning: str, tasks_html: str = "") -> str:
        """Format answer with collapsible tasks and reasoning sections."""
        formatted = answer

        if tasks_html:
            formatted += f"""

<details>
<summary>Tasks</summary>
{tasks_html}
</details>"""

        if reasoning:
            formatted += f"""

<details>
<summary>Reasoning</summary>
{reasoning}
</details>"""

        return formatted

    def _format_with_reasoning(self, answer: str, reasoning: str) -> str:
        """Format answer with collapsible reasoning section."""
        formatted = answer
        if reasoning:
            formatted += f"""

<details>
<summary>Reasoning</summary>
{reasoning}
</details>"""
        return formatted

    def _retrieve_context(self, question: str, context_type: str = "default") -> str:
        """Retrieve context based on specified type.

        Args:
            question: The question to retrieve context for
            context_type: "default" (datasource + llm), "web_search", or "datasource"
        """
        context_parts = []

        if context_type == "datasource":
            docs = self.vectorstore.similarity_search(question, k=3)
            if docs:
                context_parts.append("\n".join([doc.page_content for doc in docs]))
            print(f"📚 Datasource retrieval → {len(docs)} documents")

        if context_type == "default":
            print(f"💡 Using LLM knowledge (no additional context)")

        if context_type == "web_search":
            try:
                search_results = self.search_service._brave_search(question)
                if search_results:
                    fetched_content = self.search_service._fetch_content_from_links(search_results[:5])
                    if fetched_content:
                        formatted = self.search_service._format_search_results(fetched_content)
                        context_parts.append(formatted)
                        print(f"🌐 Web search → {len(fetched_content)} sources retrieved")
            except Exception as e:
                print(f"⚠ Web search failed (falling back to LLM): {e}")

        print(f"Final context: {context_parts}")
        context = "\n".join(context_parts)
        return context

    def _format_conversation_context(self, conversation_history: list = None) -> str:
        """Format conversation history for LLM context."""
        if not conversation_history:
            return ""

        history_text = "Previous conversation:\n"
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        return history_text

    def _invoke_llm(self, question: str, context: str = "", conversation_history: list = None) -> str:
        """Invoke LLM chain with appropriate template."""
        conversation_context = self._format_conversation_context(conversation_history)

        if context:
            chain = (
                ChatPromptTemplate.from_template(self.kb_template)
                | self.llm
                | StrOutputParser()
            )
            raw_response = chain.invoke({
                "context": context,
                "conversation_context": conversation_context,
                "question": question
            })
        else:
            chain = (
                ChatPromptTemplate.from_template(self.general_template)
                | self.llm
                | StrOutputParser()
            )
            raw_response = chain.invoke({
                "conversation_context": conversation_context,
                "question": question
            })

        return raw_response

    def _process_success(self, req: QueryRequest, answer: str, reasoning: str) -> QueryResponse:
        """Process successful query and save to database."""
        formatted_answer = self._format_with_reasoning(answer, reasoning)

        conversation_id, bot_msg = ConversationService.save_exchange(
            question=req.question,
            answer=formatted_answer,
            status=MessageStatus.SUCCESS,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=formatted_answer,
            status="success",
            response_time=0,
            created_at=bot_msg.created_at.isoformat(),
        )

    def _process_error(self, req: QueryRequest, error: str) -> QueryResponse:
        """Process failed query and save error to database."""
        conversation_id, _ = ConversationService.save_exchange(
            question=req.question,
            answer=None,
            status=MessageStatus.ERROR,
            error=error,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=None,
            status="error",
            response_time=0,
            created_at="",
            error=error,
        )

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        print(f"\nStarting RAG Query for: {req.question[:50]}...")

        try:
            # 1. Fetch conversation history if this is a follow-up question
            conversation_history = None
            if req.session_id:
                from app.config.database import SessionLocal
                from app.repositories.conversation_repository import ConversationRepository
                db = SessionLocal()
                try:
                    conv = ConversationRepository.find_by_id(db, req.session_id)
                    if conv:
                        conversation_history = [
                            {
                                "role": msg.role.value,
                                "content": msg.content
                            }
                            for msg in sorted(conv.messages, key=lambda m: m.created_at)
                        ]
                finally:
                    db.close()

            # 2. Retrieve context based on frontend-specified type
            print(f"Context type: {req.context_type}")
            context = self._retrieve_context(req.question, req.context_type)

            # 3. Invoke LLM with conversation history
            raw_response = self._invoke_llm(req.question, context, conversation_history)

            # 4. Parse response
            answer, reasoning = self._parse_response(raw_response)

            # 5. Process success and save to database
            return self._process_success(req, answer, reasoning)

        except ValueError:
            raise
        except Exception as e:
            return self._process_error(req, str(e))
