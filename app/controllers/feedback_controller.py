from flask import Blueprint, jsonify, request

from app.services.feedback_service import FeedbackService
from app.utils.logger import get_logger, log_api_call

logger = get_logger(__name__)
feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback", methods=["POST"])
@log_api_call(logger)
def submit_feedback():
    data = request.get_json()

    required_fields = ["conversation_id", "user_msg_id", "bot_msg_id", "user_question", "bot_answer", "is_positive"]
    if not data or not all(field in data for field in required_fields):
        missing = [f for f in required_fields if f not in (data or {})]
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        conversation_id = data.get("conversation_id", "").strip()
        user_msg_id = data.get("user_msg_id", "").strip()
        bot_msg_id = data.get("bot_msg_id", "").strip()
        user_question = data.get("user_question", "").strip()
        bot_answer = data.get("bot_answer", "").strip()
        is_positive = data.get("is_positive")
        reason = data.get("reason", "").strip() if data.get("reason") else None

        if not isinstance(is_positive, bool):
            return jsonify({"error": "'is_positive' must be a boolean"}), 400

        if not conversation_id or not user_msg_id or not bot_msg_id:
            return jsonify({"error": "conversation_id, user_msg_id, and bot_msg_id cannot be empty"}), 400

        logger.info(f"Submitting feedback | conversation_id={conversation_id}, user_msg_id={user_msg_id}, bot_msg_id={bot_msg_id}, is_positive={is_positive}, reason={reason}")

        feedback = FeedbackService.save_feedback(
            conversation_id=conversation_id,
            user_msg_id=user_msg_id,
            bot_msg_id=bot_msg_id,
            user_question=user_question,
            bot_answer=bot_answer,
            is_positive=is_positive,
            reason=reason,
        )

        logger.info(f"Feedback saved | feedback_id={feedback.feedback_id}")

        return jsonify({
            "status": "success",
            "feedback_id": feedback.feedback_id,
            "message": "Feedback submitted successfully",
        }), 201

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@feedback_bp.route("/feedback/<feedback_id>", methods=["GET"])
@log_api_call(logger)
def get_feedback(feedback_id: str):
    try:
        logger.debug(f"Fetching feedback | feedback_id={feedback_id}")
        feedback = FeedbackService.get_feedback(feedback_id)

        if not feedback:
            logger.warning(f"Feedback not found | feedback_id={feedback_id}")
            return jsonify({"error": "Feedback not found", "status": "error"}), 404

        logger.info(f"Retrieved feedback | feedback_id={feedback_id}")

        return jsonify({
            "status": "success",
            "feedback": {
                "feedback_id": feedback.feedback_id,
                "conversation_id": feedback.conversation_id,
                "user_msg_id": feedback.user_msg_id,
                "bot_msg_id": feedback.bot_msg_id,
                "user_question": feedback.user_question,
                "bot_answer": feedback.bot_answer,
                "is_positive": feedback.is_positive,
                "reason": feedback.reason,
                "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
            }
        }), 200

    except Exception as e:
        logger.error(f"Error fetching feedback {feedback_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@feedback_bp.route("/conversations/<conversation_id>/feedback", methods=["GET"])
@log_api_call(logger)
def get_conversation_feedback(conversation_id: str):
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)

    try:
        logger.debug(f"Fetching feedback for conversation | conversation_id={conversation_id}, limit={limit}, offset={offset}")

        feedbacks = FeedbackService.get_conversation_feedback(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

        logger.info(f"Retrieved feedback | conversation_id={conversation_id}, count={len(feedbacks)}")

        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "feedback_count": len(feedbacks),
            "feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "user_msg_id": f.user_msg_id,
                    "bot_msg_id": f.bot_msg_id,
                    "user_question": f.user_question,
                    "bot_answer": f.bot_answer,
                    "is_positive": f.is_positive,
                    "reason": f.reason,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in feedbacks
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error fetching feedback for {conversation_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@feedback_bp.route("/conversations/<conversation_id>/feedback/stats", methods=["GET"])
@log_api_call(logger)
def get_feedback_stats(conversation_id: str):
    try:
        logger.debug(f"Fetching feedback stats | conversation_id={conversation_id}")

        stats = FeedbackService.get_feedback_stats(conversation_id)

        logger.info(f"Retrieved stats | conversation_id={conversation_id}, total={stats['total']}")

        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "stats": stats
        }), 200

    except Exception as e:
        logger.error(f"Error fetching feedback stats for {conversation_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@feedback_bp.route("/feedback/<feedback_id>", methods=["DELETE"])
@log_api_call(logger)
def delete_feedback(feedback_id: str):
    try:
        logger.debug(f"Deleting feedback | feedback_id={feedback_id}")

        deleted = FeedbackService.delete_feedback(feedback_id)

        if not deleted:
            logger.warning(f"Feedback not found for delete | feedback_id={feedback_id}")
            return jsonify({"error": "Feedback not found", "status": "error"}), 404

        logger.info(f"Deleted feedback | feedback_id={feedback_id}")

        return jsonify({"status": "success", "message": "Feedback deleted successfully"}), 200

    except Exception as e:
        logger.error(f"Error deleting feedback {feedback_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500
