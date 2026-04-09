"""Agent service using LangChain create_agent with tool calling and structured output."""
from langchain_core.messages import HumanMessage #type: ignore
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.messages import SystemMessage #type: ignore
from langchain_ollama import ChatOllama
from app.tools import web_search, knowledge_base_search
from app.constants.models import LLM_MAIN_MODEL
from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.services.conversation_service import ConversationService
from app.utils.logger import get_logger
from app.middleware.middleware import trim_messages, handle_tool_errors
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver # type: ignore
from app.config.pymysql import get_pymysql_conn
from langchain_core.runnables import RunnableConfig # type: ignore

logger = get_logger(__name__)

SYSTEM_PROMPT = SystemMessage(
    content="""You are a helpful university assistant.
Use the available tools to retrieve information, then answer the user's question.

IMPORTANT: Always format your response as HTML using these elements:
- <p> for paragraphs
- <ul><li> for unordered lists
- <ol><li> for ordered/numbered lists
- <table><thead><tr><th></th></tr></thead><tbody><tr><td></td></tr></tbody></table> for tabular data like course listings
- <strong> for emphasis on labels or headings
- <br> for line breaks when needed

Never return plain text. Always wrap content in appropriate HTML tags.""")


class AgentResponse(BaseModel):
    """Structured output returned by the agent."""
    answer: str = Field(
        description="Answer formatted as HTML. Use <p>, <ul><li>, <ol><li>, <table> tags. For course/module data use a <table>. Never return plain text."
    )
    reasoning: str = Field(
        description="Brief explanation of how you arrived at the answer."
    )


class AgentService:
    """LangChain agent that selects and calls tools then returns structured output."""

    def __init__(self, model: str = LLM_MAIN_MODEL, temperature: float = 0.7):
        logger.info(f"Initializing AgentService | model={model}")
        llm = ChatOllama(model=model, temperature=temperature)

        checkpointer = PyMySQLSaver(get_pymysql_conn())
        checkpointer.setup()

        self.agent = create_agent(
            model=llm,
            tools=[web_search, knowledge_base_search],
            response_format=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
            middleware=[trim_messages, handle_tool_errors],
        )
        logger.info("AgentService initialized successfully")

    def invoke(self, question: str, context_type: str = "default", thread_id: str = "default") -> tuple[str, str]:
        """Run the agent on a question and return (answer_html, reasoning_text).

        Args:
            question: The user's question
            context_type: Hint for tool selection — "web_search", "datasource", or "default"
            thread_id: Conversation ID used to scope memory checkpointing
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        message = self._build_message(question, context_type)
        logger.info(f"Agent invoke | context_type={context_type} | thread_id={thread_id} | question={question[:60]}")

        result = self.agent.invoke({"messages": [HumanMessage(content=message)]}, config)

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
            answer, reasoning = self.invoke(req.question, req.context_type, thread_id=req.session_id or "default")
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
