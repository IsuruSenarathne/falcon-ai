import time

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.conversation_service import ConversationService


class RAGService:

    def __init__(self):
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

    def _format_with_reasoning(self, answer: str, reasoning: str) -> str:
        """Format answer with collapsible reasoning section."""
        if not reasoning:
            return answer

        return f"""{answer}

<details>
<summary>Reasoning</summary>
{reasoning}
</details>"""

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        start = time.time()
        try:
            raw_response = self.rag_chain.invoke(req.question)
            response_time = time.time() - start

            # Parse response into answer and reasoning
            answer, reasoning = self._parse_response(raw_response)
            # Format answer with collapsible reasoning section
            formatted_answer = self._format_with_reasoning(answer, reasoning)

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
