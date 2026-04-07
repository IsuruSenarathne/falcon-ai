from typing import Optional, List

from app.config.database import SessionLocal
from app.repositories.feedback_repository import FeedbackRepository
from app.models.feedback import Feedback


class FeedbackService:
    """Business logic for feedback operations."""

    @staticmethod
    def save_feedback(
        conversation_id: str,
        user_msg_id: str,
        bot_msg_id: str,
        user_question: str,
        bot_answer: str,
        is_positive: bool,
    ) -> Feedback:
        db = SessionLocal()
        try:
            feedback = FeedbackRepository.create(
                db=db,
                conversation_id=conversation_id,
                user_msg_id=user_msg_id,
                bot_msg_id=bot_msg_id,
                user_question=user_question,
                bot_answer=bot_answer,
                is_positive=is_positive,
            )
            db.commit()
            db.refresh(feedback)
            return feedback
        finally:
            db.close()

    @staticmethod
    def get_feedback(feedback_id: str) -> Optional[Feedback]:
        db = SessionLocal()
        try:
            return FeedbackRepository.find_by_id(db, feedback_id)
        finally:
            db.close()

    @staticmethod
    def get_feedback_by_bot_message(bot_msg_id: str) -> Optional[Feedback]:
        db = SessionLocal()
        try:
            return FeedbackRepository.find_by_bot_message(db, bot_msg_id)
        finally:
            db.close()

    @staticmethod
    def get_feedback_by_user_message(user_msg_id: str) -> Optional[Feedback]:
        db = SessionLocal()
        try:
            return FeedbackRepository.find_by_user_message(db, user_msg_id)
        finally:
            db.close()

    @staticmethod
    def get_conversation_feedback(
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        db = SessionLocal()
        try:
            return FeedbackRepository.find_by_conversation(
                db=db,
                conversation_id=conversation_id,
                limit=limit,
                offset=offset,
            )
        finally:
            db.close()

    @staticmethod
    def get_feedback_stats(conversation_id: str) -> dict:
        db = SessionLocal()
        try:
            total = FeedbackRepository.count_by_conversation(db, conversation_id)
            positive = FeedbackRepository.count_positive_by_conversation(db, conversation_id)
            return {
                "total": total,
                "positive": positive,
                "negative": total - positive,
                "positive_rate": (positive / total * 100) if total > 0 else 0,
            }
        finally:
            db.close()

    @staticmethod
    def delete_feedback(feedback_id: str) -> bool:
        db = SessionLocal()
        try:
            feedback = FeedbackRepository.find_by_id(db, feedback_id)
            if not feedback:
                return False
            FeedbackRepository.delete_feedback(db, feedback)
            db.commit()
            return True
        finally:
            db.close()
