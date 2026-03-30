"""
SQLAlchemy models for storing RAG conversations
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Enum
from sqlalchemy.sql import func
from datetime import datetime
import enum
import uuid
from database import Base


class ConversationStatus(str, enum.Enum):
    """Status of a conversation"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class Conversation(Base):
    """Model for storing RAG conversations"""
    __tablename__ = "conversations"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Unique identifier for the conversation (UUID)
    conversation_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Question asked
    question = Column(Text, nullable=False)
    
    # Answer provided
    answer = Column(Text, nullable=True)
    
    # Status of the query
    status = Column(Enum(ConversationStatus), default=ConversationStatus.PENDING, index=True)
    
    # Error message (if any)
    error = Column(Text, nullable=True)
    
    # User identifier (optional - for multi-user scenarios)
    user_id = Column(String(255), nullable=True, index=True)
    
    # Session identifier (to group related conversations)
    session_id = Column(String(36), nullable=True, index=True)
    
    # Response time in seconds
    response_time = Column(Float, nullable=True)
    
    # Metadata/tags (for filtering and categorization)
    extra_data = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, conversation_id={self.conversation_id}, status={self.status})>"


class ConversationSession(Base):
    """Model for grouping related conversations into sessions"""
    __tablename__ = "conversation_sessions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Unique session identifier (UUID)
    session_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # User who initiated the session
    user_id = Column(String(255), nullable=True, index=True)
    
    # Session title/description
    title = Column(String(255), nullable=True)
    
    # Session metadata
    extra_data = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ConversationSession(session_id={self.session_id}, user_id={self.user_id})>"
