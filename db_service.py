"""
Database service for managing conversations
"""

from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from models import Conversation, ConversationSession, ConversationStatus
from typing import List, Optional
import time


class ConversationService:
    """Service for managing conversation data in the database"""
    
    @staticmethod
    def save_conversation(
        db: Session,
        question: str,
        answer: str,
        status: ConversationStatus = ConversationStatus.SUCCESS,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        response_time: Optional[float] = None,
        extra_data: Optional[str] = None
    ) -> Conversation:
        """
        Save a conversation to the database.
        
        Args:
            db: Database session
            question: The question asked
            answer: The answer provided
            status: Status of the conversation
            error: Error message if any
            user_id: ID of the user
            session_id: ID of the session
            response_time: Time taken to respond in seconds
            extra_data: Additional metadata
            
        Returns:
            The created Conversation object
        """
        conversation = Conversation(
            question=question,
            answer=answer,
            status=status,
            error=error,
            user_id=user_id,
            session_id=session_id,
            response_time=response_time,
            extra_data=extra_data
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    
    @staticmethod
    def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
    
    @staticmethod
    def get_all_conversations(
        db: Session,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """
        Get all conversations with optional filters.
        
        Args:
            db: Database session
            user_id: Filter by user ID
            session_id: Filter by session ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Conversation objects
        """
        query = db.query(Conversation)
        
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        if session_id:
            query = query.filter(Conversation.session_id == session_id)
        
        return query.order_by(Conversation.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_session_conversations(db: Session, session_id: str) -> List[Conversation]:
        """Get all conversations in a session"""
        return db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.asc()).all()
    
    @staticmethod
    def create_session(
        db: Session,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        extra_data: Optional[str] = None
    ) -> ConversationSession:
        """Create a new conversation session"""
        session = ConversationSession(
            user_id=user_id,
            title=title,
            extra_data=extra_data
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID"""
        return db.query(ConversationSession).filter(
            ConversationSession.session_id == session_id
        ).first()
    
    @staticmethod
    def get_user_sessions(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ConversationSession]:
        """Get all sessions for a user"""
        return db.query(ConversationSession).filter(
            ConversationSession.user_id == user_id
        ).order_by(ConversationSession.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def delete_conversation(db: Session, conversation_id: str) -> bool:
        """Delete a conversation"""
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if conversation:
            db.delete(conversation)
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_statistics(db: Session, session_id: Optional[str] = None) -> dict:
        """
        Get statistics about conversations.
        
        Args:
            db: Database session
            session_id: Optional session ID to filter by
            
        Returns:
            Dictionary containing statistics
        """
        query = db.query(Conversation)
        
        if session_id:
            query = query.filter(Conversation.session_id == session_id)
        
        total = query.count()
        success_count = query.filter(Conversation.status == ConversationStatus.SUCCESS).count()
        error_count = query.filter(Conversation.status == ConversationStatus.ERROR).count()
        
        # Calculate average response time
        avg_response_time = db.query(
            func.avg(Conversation.response_time)
        ).filter(Conversation.status == ConversationStatus.SUCCESS)
        
        if session_id:
            avg_response_time = avg_response_time.filter(Conversation.session_id == session_id)
        
        avg_response_time = avg_response_time.scalar() or 0
        
        return {
            "total_conversations": total,
            "successful": success_count,
            "errors": error_count,
            "success_rate": (success_count / total * 100) if total > 0 else 0,
            "average_response_time": float(avg_response_time)
        }
