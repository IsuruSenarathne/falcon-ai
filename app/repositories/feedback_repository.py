from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.feedback import Feedback


class FeedbackRepository:
    """Pure data-access layer for feedback — no business logic."""

    @staticmethod
    def create(
        db: Session,
        conversation_id: str,
        user_msg_id: str,
        bot_msg_id: str,
        user_question: str,
        bot_answer: str,
        is_positive: bool,
        reason: str = None,
    ) -> Feedback:
        feedback = Feedback(
            conversation_id=conversation_id,
            user_msg_id=user_msg_id,
            bot_msg_id=bot_msg_id,
            user_question=user_question,
            bot_answer=bot_answer,
            is_positive=is_positive,
            reason=reason,
        )
        db.add(feedback)
        db.flush()
        return feedback

    @staticmethod
    def find_by_id(db: Session, feedback_id: str) -> Optional[Feedback]:
        return db.query(Feedback).filter(
            Feedback.feedback_id == feedback_id
        ).first()

    @staticmethod
    def find_by_bot_message(db: Session, bot_msg_id: str) -> Optional[Feedback]:
        return db.query(Feedback).filter(
            Feedback.bot_msg_id == bot_msg_id
        ).first()

    @staticmethod
    def find_by_user_message(db: Session, user_msg_id: str) -> Optional[Feedback]:
        return db.query(Feedback).filter(
            Feedback.user_msg_id == user_msg_id
        ).first()

    @staticmethod
    def find_by_conversation(
        db: Session,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        return db.query(Feedback).filter(
            Feedback.conversation_id == conversation_id
        ).order_by(Feedback.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def count_by_conversation(db: Session, conversation_id: str) -> int:
        return db.query(Feedback).filter(
            Feedback.conversation_id == conversation_id
        ).count()

    @staticmethod
    def count_positive_by_conversation(db: Session, conversation_id: str) -> int:
        return db.query(Feedback).filter(
            Feedback.conversation_id == conversation_id,
            Feedback.is_positive == True,
        ).count()

    @staticmethod
    def delete_feedback(db: Session, feedback: Feedback) -> None:
        db.delete(feedback)
        db.flush()
