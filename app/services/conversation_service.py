import uuid
from typing import Optional, Tuple

from app.config.database import SessionLocal
from app.dto.conversation_dto import (
    ConversationDTO,
    ConversationResponse,
    ConversationsListResponse,
    MessageDTO,
    MessagesResponse,
    MessageResponse,
)
from app.models.conversation import Message, MessageRole, MessageStatus
from app.repositories.conversation_repository import ConversationRepository


class ConversationService:

    @staticmethod
    def save_exchange(
        question: str,
        answer: Optional[str],
        status: MessageStatus,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        response_time: Optional[float] = None,
    ) -> Tuple[str, Message]:
        """
        Persist a user question + bot answer as two Messages inside a Conversation.
        Creates the Conversation if it doesn't exist (using session_id as its ID).
        Returns (conversation_id, bot_message).
        """
        db = SessionLocal()
        try:
            conversation_id = session_id or str(uuid.uuid4())

            conversation = ConversationRepository.find_by_id(db, conversation_id)
            if not conversation:
                conversation = ConversationRepository.create(db, conversation_id, user_id)

            ConversationRepository.add_message(db, conversation_id, MessageRole.USER, question)
            bot_msg = ConversationRepository.add_message(
                db,
                conversation_id,
                MessageRole.BOT,
                content=answer or "",
                status=status,
                error=error,
                response_time=response_time,
            )

            db.commit()
            db.refresh(bot_msg)
            return conversation_id, bot_msg
        finally:
            db.close()

    @staticmethod
    def get_conversations(
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ConversationsListResponse:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        db = SessionLocal()
        try:
            conversations = ConversationRepository.find_all(db, user_id, limit, offset)
            total = ConversationRepository.count(db, user_id)

            items = [
                ConversationDTO(
                    conversation_id=str(conv.conversation_id),
                    user_id=conv.user_id,
                    title=conv.title,
                    messages=[
                        MessageDTO(
                            message_id=str(msg.message_id),
                            role=msg.role.value,
                            content=msg.content,
                            status=msg.status.value if msg.status else None,
                            error=msg.error,
                            response_time=msg.response_time,
                            created_at=msg.created_at.isoformat() if msg.created_at else None,
                        )
                        for msg in sorted(conv.messages, key=lambda m: m.created_at)
                    ],
                    created_at=conv.created_at.isoformat() if conv.created_at else None,
                )
                for conv in conversations
            ]

            return ConversationsListResponse(
                conversations=items, total=total, limit=limit, offset=offset
            )
        finally:
            db.close()

    @staticmethod
    def get_conversation(conversation_id: str) -> Optional[ConversationResponse]:
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                return None
            dto = ConversationDTO(
                conversation_id=str(conv.conversation_id),
                user_id=conv.user_id,
                title=conv.title,
                messages=[
                    MessageDTO(
                        message_id=str(msg.message_id),
                        role=msg.role.value,
                        content=msg.content,
                        status=msg.status.value if msg.status else None,
                        error=msg.error,
                        response_time=msg.response_time,
                        created_at=msg.created_at.isoformat() if msg.created_at else None,
                    )
                    for msg in sorted(conv.messages, key=lambda m: m.created_at)
                ],
                created_at=conv.created_at.isoformat() if conv.created_at else None,
            )
            return ConversationResponse(conversation=dto)
        finally:
            db.close()

    @staticmethod
    def update_conversation(conversation_id: str, title: Optional[str]) -> Optional[ConversationResponse]:
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                return None
            updated = ConversationRepository.update_conversation(db, conv, title)
            db.commit()
            db.refresh(updated)
            dto = ConversationDTO(
                conversation_id=str(updated.conversation_id),
                user_id=updated.user_id,
                title=updated.title,
                messages=[
                    MessageDTO(
                        message_id=str(msg.message_id),
                        role=msg.role.value,
                        content=msg.content,
                        status=msg.status.value if msg.status else None,
                        error=msg.error,
                        response_time=msg.response_time,
                        created_at=msg.created_at.isoformat() if msg.created_at else None,
                    )
                    for msg in sorted(updated.messages, key=lambda m: m.created_at)
                ],
                created_at=updated.created_at.isoformat() if updated.created_at else None,
            )
            return ConversationResponse(conversation=dto)
        finally:
            db.close()

    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                return False
            ConversationRepository.delete_conversation(db, conv)
            db.commit()
            return True
        finally:
            db.close()

    @staticmethod
    def get_message(conversation_id: str, message_id: str) -> Optional[MessageResponse]:
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                return None
            return MessageResponse(
                conversation_id=conversation_id,
                message=MessageDTO(
                    message_id=str(msg.message_id),
                    role=msg.role.value,
                    content=msg.content,
                    status=msg.status.value if msg.status else None,
                    error=msg.error,
                    response_time=msg.response_time,
                    created_at=msg.created_at.isoformat() if msg.created_at else None,
                ),
            )
        finally:
            db.close()

    @staticmethod
    def update_message(conversation_id: str, message_id: str, content: str) -> Optional[MessageResponse]:
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                return None
            updated = ConversationRepository.update_message_content(db, msg, content)
            db.commit()
            db.refresh(updated)
            return MessageResponse(
                conversation_id=conversation_id,
                message=MessageDTO(
                    message_id=str(updated.message_id),
                    role=updated.role.value,
                    content=updated.content,
                    status=updated.status.value if updated.status else None,
                    error=updated.error,
                    response_time=updated.response_time,
                    created_at=updated.created_at.isoformat() if updated.created_at else None,
                ),
            )
        finally:
            db.close()

    @staticmethod
    def delete_message(conversation_id: str, message_id: str) -> bool:
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                return False
            ConversationRepository.delete_message(db, msg)
            db.commit()
            return True
        finally:
            db.close()

    @staticmethod
    def get_messages(conversation_id: str) -> Optional[MessagesResponse]:
        db = SessionLocal()
        try:
            conversation = ConversationRepository.find_by_id(db, conversation_id)
            if not conversation:
                return None

            messages = [
                MessageDTO(
                    message_id=str(msg.message_id),
                    role=msg.role.value,
                    content=msg.content,
                    status=msg.status.value if msg.status else None,
                    error=msg.error,
                    response_time=msg.response_time,
                    created_at=msg.created_at.isoformat() if msg.created_at else None,
                )
                for msg in sorted(conversation.messages, key=lambda m: m.created_at)
            ]

            return MessagesResponse(conversation_id=conversation_id, messages=messages)
        finally:
            db.close()
