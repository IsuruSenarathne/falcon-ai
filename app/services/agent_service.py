"""Agent service using LangChain create_agent with tool calling and structured output."""
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from app.tools import web_search, knowledge_base_search
from app.constants.models import LLM_MAIN_MODEL
from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.services.conversation_service import ConversationService
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful university assistant.
answer the user's question using tools.

"""


class AgentResponse(BaseModel):
    """Structured output returned by the agent."""
    answer: str = Field(
        description="HTML formatted answer using <p>, <ul><li>, <ol><li>, or <table> tags"
    )
    reasoning: str = Field(
        description="your reasoning here..."
    )


class AgentService:
    """LangChain agent that selects and calls tools then returns structured output."""

    def __init__(self, model: str = LLM_MAIN_MODEL, temperature: float = 0.7):
        logger.info(f"Initializing AgentService | model={model}")
        llm = ChatOllama(model=model, temperature=temperature)

        self.agent = create_agent(
            model=llm,
            tools=[web_search, knowledge_base_search],
            response_format=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
        )
        logger.info("AgentService initialized successfully")

    def invoke(self, question: str, context_type: str = "default") -> tuple[str, str]:
        """Run the agent on a question and return (answer_html, reasoning_text).

        Args:
            question: The user's question
            context_type: Hint for tool selection — "web_search", "datasource", or "default"
        """
        message = self._build_message(question, context_type)
        logger.info(f"Agent invoke | context_type={context_type} | question={question[:60]}")

        result = self.agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })

        structured: AgentResponse = result.get("structured_response")
        if structured:
            logger.debug(f"Structured response received | answer_len={len(structured.answer)}")
            return structured.answer, structured.reasoning

        logger.warning("No structured_response — extracting from messages")
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") != "tool":
                return f"<p>{msg.content}</p>", ""
        return "<p>No response generated.</p>", ""

    def query(self, req: QueryRequest) -> QueryResponse:
        """Entry point for controllers — runs the agent and persists the exchange to DB."""
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        try:
            answer, reasoning = self.invoke(req.question, req.context_type)
            db_answer = self._combine_for_storage(answer, reasoning)

            conversation_id, bot_msg = ConversationService.save_exchange(
                question=req.question,
                answer=db_answer,
                status=MessageStatus.SUCCESS,
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=0,
            )
            return QueryResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=answer,
                reasoning=reasoning or None,
                status="success",
                response_time=0,
                created_at=bot_msg.created_at.isoformat(),
            )
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            conversation_id, _ = ConversationService.save_exchange(
                question=req.question,
                answer=None,
                status=MessageStatus.ERROR,
                error=str(e),
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
                error=str(e),
            )

    def _combine_for_storage(self, answer: str, reasoning: str) -> str:
        parts = [answer] if answer else []
        if reasoning:
            lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
            items = "".join(f"<li>{l}</li>" for l in lines)
            parts.append(f"<p><strong>Reasoning:</strong></p><ol>{items}</ol>")
        return "".join(parts) or "<p>No response generated.</p>"

    def _build_message(self, question: str, context_type: str) -> str:
        """Prepend a tool hint to guide the agent when context_type is explicit."""
        if context_type == "web_search":
            return f"Use provided context from web search to answer: {question}"
        if context_type == "datasource":
            return f"Search the provided context as knowledge base to answer: {question}"
        return question
