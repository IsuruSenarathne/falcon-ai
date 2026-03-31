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

Respond in two sections:
1. ANSWER: Provide a clear, direct answer in valid HTML.
2. REASONING: Explain your reasoning and how you derived the answer from the context.

Format your response exactly like this:
ANSWER:
<p>Your answer here...</p>

REASONING:
<p>Your reasoning here...</p>

Rules:
- Always respond with valid HTML using whatever tags best suit the content.
- Do NOT include diagrams, charts, or images.
- Do NOT wrap the response in <html>, <head>, <body>, <style>, or <script> tags — return inner HTML content only.
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

        if "ANSWER:" in raw_response and "REASONING:" in raw_response:
            parts = raw_response.split("REASONING:")
            answer_part = parts[0].replace("ANSWER:", "").strip()
            reasoning_part = parts[1].strip() if len(parts) > 1 else ""

            answer = answer_part
            reasoning = reasoning_part
        else:
            # Fallback if parsing fails
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
