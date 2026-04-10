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
from app.utils.logger import get_logger, log_service_call, log_errors

logger = get_logger(__name__)


class ConversationService:

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
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
            logger.debug(f"Saving exchange | conversation_id={conversation_id}, user_id={user_id}, status={status.value}")

            conversation = ConversationRepository.find_by_id(db, conversation_id)
            if not conversation:
                logger.debug(f"Creating new conversation | id={conversation_id}")
                conversation = ConversationRepository.create(db, conversation_id, user_id)

            ConversationRepository.add_message(db, conversation_id, MessageRole.USER, question)
            logger.debug(f"Added user message | conversation_id={conversation_id}")

            bot_msg = ConversationRepository.add_message(
                db,
                conversation_id,
                MessageRole.BOT,
                content=answer or "",
                status=status,
                error=error,
                response_time=response_time,
            )
            logger.debug(f"Added bot message | conversation_id={conversation_id}")

            db.commit()
            db.refresh(bot_msg)
            logger.info(f"Exchange saved successfully | conversation_id={conversation_id}")
            return conversation_id, bot_msg
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def get_conversations(
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ConversationsListResponse:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        logger.debug(f"Fetching conversations | user_id={user_id}, limit={limit}, offset={offset}")

        db = SessionLocal()
        try:
            conversations = ConversationRepository.find_all(db, user_id, limit, offset)
            total = ConversationRepository.count(db, user_id)
            logger.debug(f"Retrieved {len(conversations)} conversations from DB | total={total}")

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

            logger.info(f"Conversations list built | count={len(items)}")
            return ConversationsListResponse(
                conversations=items, total=total, limit=limit, offset=offset
            )
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def get_conversation(conversation_id: str) -> Optional[ConversationResponse]:
        logger.debug(f"Fetching conversation | id={conversation_id}")
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                logger.warning(f"Conversation not found | id={conversation_id}")
                return None

            logger.debug(f"Building conversation DTO | id={conversation_id}, messages={len(conv.messages)}")
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
            logger.info(f"Conversation retrieved | id={conversation_id}")
            return ConversationResponse(conversation=dto)
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def update_conversation(conversation_id: str, title: Optional[str]) -> Optional[ConversationResponse]:
        logger.debug(f"Updating conversation | id={conversation_id}, title={title}")
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                logger.warning(f"Conversation not found for update | id={conversation_id}")
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
            logger.info(f"Conversation updated | id={conversation_id}")
            return ConversationResponse(conversation=dto)
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def delete_conversation(conversation_id: str) -> bool:
        logger.debug(f"Deleting conversation | id={conversation_id}")
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                logger.warning(f"Conversation not found for delete | id={conversation_id}")
                return False

            ConversationRepository.delete_conversation(db, conv)
            db.commit()
            logger.info(f"Conversation deleted | id={conversation_id}")
            return True
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def get_message(conversation_id: str, message_id: str) -> Optional[MessageResponse]:
        logger.debug(f"Fetching message | conversation_id={conversation_id}, message_id={message_id}")
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                logger.warning(f"Message not found | message_id={message_id}")
                return None

            logger.info(f"Message retrieved | conversation_id={conversation_id}, message_id={message_id}")
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
    @log_service_call(logger)
    @log_errors(logger)
    def update_message(conversation_id: str, message_id: str, content: str) -> Optional[MessageResponse]:
        logger.debug(f"Updating message | conversation_id={conversation_id}, message_id={message_id}")
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                logger.warning(f"Message not found for update | message_id={message_id}")
                return None

            updated = ConversationRepository.update_message_content(db, msg, content)
            db.commit()
            db.refresh(updated)

            logger.info(f"Message updated | conversation_id={conversation_id}, message_id={message_id}")
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
    @log_service_call(logger)
    @log_errors(logger)
    def delete_message(conversation_id: str, message_id: str) -> bool:
        logger.debug(f"Deleting message | conversation_id={conversation_id}, message_id={message_id}")
        db = SessionLocal()
        try:
            msg = ConversationRepository.find_message(db, conversation_id, message_id)
            if not msg:
                logger.warning(f"Message not found for delete | message_id={message_id}")
                return False

            ConversationRepository.delete_message(db, msg)
            db.commit()
            logger.info(f"Message deleted | conversation_id={conversation_id}, message_id={message_id}")
            return True
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def get_conversation_history(conversation_id: str) -> list:
        """Get conversation history as a list of message dicts with role and content."""
        logger.debug(f"Fetching conversation history | id={conversation_id}")
        db = SessionLocal()
        try:
            conv = ConversationRepository.find_by_id(db, conversation_id)
            if not conv:
                logger.warning(f"Conversation not found | id={conversation_id}")
                return []

            history = [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in sorted(conv.messages, key=lambda m: m.created_at)
            ]
            logger.info(f"Conversation history retrieved | id={conversation_id}, messages={len(history)}")
            return history
        finally:
            db.close()

    @staticmethod
    @log_service_call(logger)
    @log_errors(logger)
    def get_messages(conversation_id: str) -> Optional[MessagesResponse]:
        logger.debug(f"Fetching messages | conversation_id={conversation_id}")
        db = SessionLocal()
        try:
            conversation = ConversationRepository.find_by_id(db, conversation_id)
            if not conversation:
                logger.warning(f"Conversation not found | id={conversation_id}")
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

            logger.info(f"Messages retrieved | conversation_id={conversation_id}, count={len(messages)}")
            return MessagesResponse(conversation_id=conversation_id, messages=messages)
        finally:
            db.close()
