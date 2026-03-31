from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message, MessageRole, MessageStatus


class ConversationRepository:
    """Pure data-access layer — no business logic."""

    @staticmethod
    def find_by_id(db: Session, conversation_id: str) -> Optional[Conversation]:
        return db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()

    @staticmethod
    def create(db: Session, conversation_id: str, user_id: Optional[str] = None) -> Conversation:
        conversation = Conversation(conversation_id=conversation_id, user_id=user_id)
        db.add(conversation)
        db.flush()
        return conversation

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: str,
        role: MessageRole,
        content: str,
        status: Optional[MessageStatus] = None,
        error: Optional[str] = None,
        response_time: Optional[float] = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            status=status,
            error=error,
            response_time=response_time,
        )
        db.add(message)
        return message

    @staticmethod
    def find_all(
        db: Session,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Conversation]:
        query = db.query(Conversation)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        return query.order_by(Conversation.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def count(db: Session, user_id: Optional[str] = None) -> int:
        query = db.query(Conversation)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        return query.count()
