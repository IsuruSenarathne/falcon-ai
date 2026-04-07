import enum
import uuid

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.config.database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    feedback_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), nullable=False, index=True)
    user_msg_id = Column(String(255), nullable=False, index=True)
    bot_msg_id = Column(String(255), nullable=False, index=True)
    user_question = Column(Text, nullable=False)
    bot_answer = Column(Text, nullable=False)
    is_positive = Column(Boolean, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    conversation = relationship("Conversation")

    def __repr__(self):
        return f"<Feedback(feedback_id={self.feedback_id}, is_positive={self.is_positive})>"
