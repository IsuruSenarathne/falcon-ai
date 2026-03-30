"""
Core RAG Service Module
Handles all business logic for the RAG (Retrieval-Augmented Generation) system.
Stores conversations in the database.
"""

import os
import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from database import SessionLocal
from db_service import ConversationService
from models import ConversationStatus


class RAGService:
    """Service class for RAG chain operations."""
    
    def __init__(self, data_file: str = "data.json"):
        """
        Initialize the RAG service.
        
        Args:
            data_file: Path to the JSON file containing the knowledge base.
        """
        load_dotenv()
        
        # Setup API Key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Load data from JSON file
        with open(data_file, "r") as f:
            data = json.load(f)["data"]
        
        # Create embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Create knowledge base
        knowledge_base = [f"Course: {title}. {desc}" for title, desc in data.items()]
        
        # Create vector store and retriever
        self.vectorstore = Chroma.from_texts(texts=knowledge_base, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        # Define the RAG Prompt
        template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize Model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        # Create the RAG Chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, user_id: str = None, session_id: str = None) -> dict:
        """
        Process a single question through the RAG chain and store in database.
        
        Args:
            question: The question to answer.
            user_id: Optional user identifier for tracking.
            session_id: Optional session identifier for grouping conversations.
            
        Returns:
            Dictionary containing conversation_id, question, answer, and metadata.
            
        Raises:
            ValueError: If the question is empty.
            Exception: If there's an error processing the query.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        db = SessionLocal()
        start_time = time.time()
        
        try:
            # Get answer from RAG chain
            answer = self.rag_chain.invoke(question)
            response_time = time.time() - start_time
            
            # Save conversation to database
            conversation = ConversationService.save_conversation(
                db=db,
                question=question,
                answer=answer,
                status=ConversationStatus.SUCCESS,
                user_id=user_id,
                session_id=session_id,
                response_time=response_time
            )
            
            return {
                "conversation_id": conversation.conversation_id,
                "question": conversation.question,
                "answer": conversation.answer,
                "status": "success",
                "response_time": response_time,
                "created_at": conversation.created_at.isoformat()
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            
            # Save error to database
            try:
                conversation = ConversationService.save_conversation(
                    db=db,
                    question=question,
                    answer=None,
                    status=ConversationStatus.ERROR,
                    error=str(e),
                    user_id=user_id,
                    session_id=session_id,
                    response_time=response_time
                )
                return {
                    "conversation_id": conversation.conversation_id,
                    "question": question,
                    "answer": None,
                    "error": str(e),
                    "status": "error",
                    "response_time": response_time
                }
            except:
                raise
        
        finally:
            db.close()
    
    def batch_query(self, questions: list, user_id: str = None, session_id: str = None) -> dict:
        """
        Process multiple questions through the RAG chain and store in database.
        
        Args:
            questions: List of questions to answer.
            user_id: Optional user identifier for tracking.
            session_id: Optional session identifier for grouping conversations.
            
        Returns:
            Dictionary containing list of results with conversation details.
            
        Raises:
            ValueError: If questions list is empty or not a list.
        """
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")
        
        if not questions:
            raise ValueError("Questions list cannot be empty")
        
        db = SessionLocal()
        results = []
        
        try:
            for question in questions:
                start_time = time.time()
                
                if not question or not question.strip():
                    # Save empty question error
                    conversation = ConversationService.save_conversation(
                        db=db,
                        question=question,
                        answer=None,
                        status=ConversationStatus.ERROR,
                        error="Question cannot be empty",
                        user_id=user_id,
                        session_id=session_id,
                        response_time=0
                    )
                    results.append({
                        "conversation_id": conversation.conversation_id,
                        "question": question,
                        "answer": None,
                        "error": "Question cannot be empty",
                        "status": "error"
                    })
                else:
                    try:
                        # Get answer from RAG chain
                        answer = self.rag_chain.invoke(question)
                        response_time = time.time() - start_time
                        
                        # Save successful conversation
                        conversation = ConversationService.save_conversation(
                            db=db,
                            question=question,
                            answer=answer,
                            status=ConversationStatus.SUCCESS,
                            user_id=user_id,
                            session_id=session_id,
                            response_time=response_time
                        )
                        
                        results.append({
                            "conversation_id": conversation.conversation_id,
                            "question": question,
                            "answer": answer,
                            "status": "success",
                            "response_time": response_time
                        })
                    except Exception as e:
                        response_time = time.time() - start_time
                        
                        # Save error conversation
                        conversation = ConversationService.save_conversation(
                            db=db,
                            question=question,
                            answer=None,
                            status=ConversationStatus.ERROR,
                            error=str(e),
                            user_id=user_id,
                            session_id=session_id,
                            response_time=response_time
                        )
                        
                        results.append({
                            "conversation_id": conversation.conversation_id,
                            "question": question,
                            "answer": None,
                            "error": str(e),
                            "status": "error",
                            "response_time": response_time
                        })
            
            return {
                "results": results,
                "session_id": session_id,
                "total": len(results),
                "status": "success"
            }
        
        finally:
            db.close()
