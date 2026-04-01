import time

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.dto.conversation_dto import QueryRequest, QueryResponse, TaskBreakdownRequest
from app.models.conversation import MessageStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.conversation_service import ConversationService


class RAGService:

    def __init__(self, task_breakdown_service=None):
        self.task_breakdown_service = task_breakdown_service
        documents = KnowledgeRepository.load_documents()
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
        print(f"✓ Knowledge base loaded: {len(documents)} documents from database")

        template = """Answer the question based ONLY on the following context:
{context}

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
        self.rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | ChatOllama(model="llama3.2:1b")
            | StrOutputParser()
        )

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
        return self._format_with_tasks_and_reasoning(answer, reasoning, "")

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        start = time.time()
        try:
            # Break down the question into tasks for better understanding
            task_context = ""
            tasks_list = []
            tasks_html = ""

            if self.task_breakdown_service:
                try:
                    breakdown_req = TaskBreakdownRequest(statement=req.question)
                    breakdown = self.task_breakdown_service.breakdown(breakdown_req)
                    if breakdown.status == "success" and breakdown.tasks:
                        tasks_list = breakdown.tasks
                        task_list = "\n".join([f"- {t.title}: {t.description}" for t in breakdown.tasks])
                        task_context = f"Key tasks to address:\n{task_list}"
                        # Format tasks as HTML for display
                        tasks_html = "<ul>\n" + "\n".join([
                            f"<li><strong>{t.title}</strong> ({t.priority}): {t.description}</li>"
                            for t in breakdown.tasks
                        ]) + "\n</ul>"
                except Exception as e:
                    print(f"Task breakdown failed (non-critical): {e}")

            # Prepare enhanced question with task context
            enhanced_question = req.question
            if task_context:
                enhanced_question = f"{req.question}\n\n{task_context}"

            raw_response = self.rag_chain.invoke(enhanced_question)
            response_time = time.time() - start

            # Parse response into answer and reasoning
            answer, reasoning = self._parse_response(raw_response)
            # Format answer with collapsible tasks and reasoning sections
            formatted_answer = self._format_with_tasks_and_reasoning(answer, reasoning, tasks_html)

            conversation_id, bot_msg = ConversationService.save_exchange(
                question=req.question,
                answer=formatted_answer,
                status=MessageStatus.SUCCESS,
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=response_time,
            )

            return QueryResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=formatted_answer,
                status="success",
                response_time=response_time,
                created_at=bot_msg.created_at.isoformat(),
                tasks=tasks_list if tasks_list else None,
            )

        except ValueError:
            raise
        except Exception as e:
            response_time = time.time() - start

            conversation_id, _ = ConversationService.save_exchange(
                question=req.question,
                answer=None,
                status=MessageStatus.ERROR,
                error=str(e),
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=response_time,
            )

            return QueryResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=None,
                status="error",
                response_time=response_time,
                created_at="",
                error=str(e),
            )
