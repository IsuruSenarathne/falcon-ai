"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ConversationRequest(BaseModel):
    """Schema for incoming conversation requests"""
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Optional[str] = None


class ConversationResponse(BaseModel):
    """Schema for conversation responses"""
    conversation_id: str
    question: str
    answer: str
    status: str
    error: Optional[str] = None
    response_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class BatchQueryRequest(BaseModel):
    """Schema for batch query requests"""
    questions: List[str]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class BatchQueryResponse(BaseModel):
    """Schema for batch query responses"""
    results: List[ConversationResponse]
    status: str
    session_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Schema for creating a new session"""
    user_id: Optional[str] = None
    title: Optional[str] = None
    extra_data: Optional[str] = None


class SessionResponse(BaseModel):
    """Schema for session response"""
    session_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Schema for statistics response"""
    total_conversations: int
    successful: int
    errors: int
    success_rate: float
    average_response_time: float
