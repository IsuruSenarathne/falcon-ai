"""
Database service for managing conversations and messages.
"""

from sqlalchemy.orm import Session
from models import Conversation, Message, MessageRole, MessageStatus
from typing import Optional
import uuid


class ConversationService:
    """Service for managing conversation/message data in the database."""

    @staticmethod
    def save_conversation(
        db: Session,
        question: str,
        answer: Optional[str],
        status: MessageStatus = MessageStatus.SUCCESS,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        response_time: Optional[float] = None,
        extra_data: Optional[str] = None
    ) -> Message:
        """
        Save a question/answer pair as two messages inside a conversation.

        Uses session_id as the conversation_id. Creates the conversation if it doesn't exist.
        Returns the bot Message object.
        """
        # Find or create conversation
        conversation = None
        if session_id:
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == session_id
            ).first()
        if not conversation:
            conversation = Conversation(
                conversation_id=session_id or str(uuid.uuid4()),
                user_id=user_id
            )
            db.add(conversation)
            db.flush()

        # Save user message
        user_msg = Message(
            conversation_id=conversation.conversation_id,
            role=MessageRole.USER,
            content=question
        )
        db.add(user_msg)

        # Save bot message
        bot_msg = Message(
            conversation_id=conversation.conversation_id,
            role=MessageRole.BOT,
            content=answer or "",
            status=status,
            error=error,
            response_time=response_time
        )
        db.add(bot_msg)
        db.commit()
        db.refresh(bot_msg)
        return bot_msg

    @staticmethod
    def get_conversations_list(
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> dict:
        """
        Retrieve conversations with their messages.
        """
        if limit < 1 or limit > 1000:
            limit = 100
        if offset < 0:
            offset = 0

        from database import SessionLocal
        db = SessionLocal()

        try:
            query = db.query(Conversation)

            if session_id:
                query = query.filter(Conversation.conversation_id == session_id)
            if user_id:
                query = query.filter(Conversation.user_id == user_id)

            total_count = query.count()
            conversations = query.order_by(Conversation.created_at.desc()).offset(offset).limit(limit).all()

            result = []
            for conv in conversations:
                messages = [
                    {
                        "message_id": str(msg.message_id),
                        "role": msg.role.value,
                        "content": msg.content,
                        "status": msg.status.value if msg.status else None,
                        "error": msg.error,
                        "response_time": msg.response_time,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None
                    }
                    for msg in sorted(conv.messages, key=lambda m: m.created_at)
                ]
                result.append({
                    "conversation_id": str(conv.conversation_id),
                    "user_id": conv.user_id,
                    "title": conv.title,
                    "messages": messages,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None
                })

            return {
                "conversations": result,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "status": "success"
            }

        finally:
            db.close()

    @staticmethod
    def get_messages(conversation_id: str) -> dict:
        """
        Retrieve all messages for a specific conversation, ordered by created_at asc.
        """
        from database import SessionLocal
        db = SessionLocal()

        try:
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id
            ).first()

            if not conversation:
                return {"error": "Conversation not found", "status": "error"}

            messages = sorted(conversation.messages, key=lambda m: m.created_at)

            return {
                "conversation_id": conversation_id,
                "messages": [
                    {
                        "message_id": str(msg.message_id),
                        "role": msg.role.value,
                        "content": msg.content,
                        "status": msg.status.value if msg.status else None,
                        "error": msg.error,
                        "response_time": msg.response_time,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None
                    }
                    for msg in messages
                ],
                "status": "success"
            }

        finally:
            db.close()
