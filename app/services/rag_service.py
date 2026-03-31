import json
import time

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.services.conversation_service import ConversationService


class RAGService:

    def __init__(self, data_file: str = "data.json"):
        with open(data_file, "r") as f:
            data = json.load(f)["data"]

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        knowledge_base = [f"Course: {title}. {desc}" for title, desc in data.items()]
        vectorstore = Chroma.from_texts(texts=knowledge_base, embedding=embeddings)

        template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
        self.rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | ChatOllama(model="llama3.2:1b")
            | StrOutputParser()
        )

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        start = time.time()
        try:
            answer = self.rag_chain.invoke(req.question)
            response_time = time.time() - start

            conversation_id, bot_msg = ConversationService.save_exchange(
                question=req.question,
                answer=answer,
                status=MessageStatus.SUCCESS,
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=response_time,
            )

            return QueryResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=answer,
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
