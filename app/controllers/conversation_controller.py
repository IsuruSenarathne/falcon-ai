import uuid

from flask import Blueprint, current_app, jsonify, request

from app.dto.conversation_dto import QueryRequest
from app.services.conversation_service import ConversationService
from app.utils.logger import get_logger, log_api_call

logger = get_logger(__name__)
conversation_bp = Blueprint("conversations", __name__)


@conversation_bp.route("/", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "AI Nemo"})


@conversation_bp.route("/query", methods=["POST"])
@log_api_call(logger)
def query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data.get("question", "")
        context_type = data.get("context_type", "default")
        user_id = data.get("user_id", "anonymous")

        logger.info(f"Query received | user_id={user_id}, context_type={context_type}")
        logger.debug(f"Question: {question[:50]}...")

        req = QueryRequest.from_json(data)
        result = current_app.agent_service.query(req)

        logger.info(f"Query result | status={result.status}")

        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations", methods=["POST"])
@log_api_call(logger)
def create_conversation():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        session_id = str(uuid.uuid4())
        logger.debug(f"Creating new conversation | session_id={session_id}")

        session_data = {**data, "session_id": session_id}
        req = QueryRequest.from_json(session_data)
        result = current_app.agent_service.query(req)

        logger.info(f"Conversation created | id={result.conversation_id}")
        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        logger.warning(f"Validation error in create_conversation: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations", methods=["GET"])
@log_api_call(logger)
def get_conversations():
    user_id = request.args.get("user_id")
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)

    try:
        logger.debug(f"Fetching conversations | user_id={user_id}, limit={limit}, offset={offset}")
        result = ConversationService.get_conversations(user_id=user_id, limit=limit, offset=offset)
        logger.info(f"Retrieved {len(result.conversations)} conversations | total={result.total}")
        return jsonify(result.to_dict()), 200
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>", methods=["GET", "PATCH", "DELETE"])
@log_api_call(logger)
def conversation(conversation_id: str):
    if request.method == "GET":
        try:
            logger.debug(f"Fetching conversation | id={conversation_id}")
            result = ConversationService.get_conversation(conversation_id)
            if result is None:
                logger.warning(f"Conversation not found | id={conversation_id}")
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            logger.info(f"Retrieved conversation | id={conversation_id}, messages={len(result.conversation.messages)}")
            return jsonify(result.to_dict()), 200
        except Exception as e:
            logger.error(f"Error fetching conversation {conversation_id}: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    if request.method == "PATCH":
        data = request.get_json() or {}
        if "title" not in data:
            return jsonify({"error": "Missing 'title' field"}), 400
        try:
            logger.debug(f"Updating conversation | id={conversation_id}, title={data['title']}")
            result = ConversationService.update_conversation(conversation_id, data["title"])
            if result is None:
                logger.warning(f"Conversation not found for update | id={conversation_id}")
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            logger.info(f"Updated conversation | id={conversation_id}")
            return jsonify(result.to_dict()), 200
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    # DELETE
    try:
        logger.debug(f"Deleting conversation | id={conversation_id}")
        deleted = ConversationService.delete_conversation(conversation_id)
        if not deleted:
            logger.warning(f"Conversation not found for delete | id={conversation_id}")
            return jsonify({"error": "Conversation not found", "status": "error"}), 404
        logger.info(f"Deleted conversation | id={conversation_id}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>/messages", methods=["GET", "POST"])
@log_api_call(logger)
def messages(conversation_id: str):
    if request.method == "GET":
        try:
            logger.debug(f"Fetching messages | conversation_id={conversation_id}")
            result = ConversationService.get_messages(conversation_id)
            if result is None:
                logger.warning(f"Conversation not found | id={conversation_id}")
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            logger.info(f"Retrieved messages | conversation_id={conversation_id}, count={len(result.messages)}")
            return jsonify(result.to_dict()), 200
        except Exception as e:
            logger.error(f"Error fetching messages for {conversation_id}: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    # POST — append a new message to an existing conversation
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        logger.debug(f"Adding message to conversation | conversation_id={conversation_id}")
        session_data = {**data, "session_id": conversation_id}
        req = QueryRequest.from_json(session_data)
        result = current_app.agent_service.query(req)

        logger.info(f"Message added | conversation_id={conversation_id}, status={result.status}")
        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        logger.warning(f"Validation error in add message: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error adding message to {conversation_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>/messages/<message_id>", methods=["GET", "PATCH", "DELETE"])
@log_api_call(logger)
def message(conversation_id: str, message_id: str):
    if request.method == "GET":
        try:
            logger.debug(f"Fetching message | conversation_id={conversation_id}, message_id={message_id}")
            result = ConversationService.get_message(conversation_id, message_id)
            if result is None:
                logger.warning(f"Message not found | conversation_id={conversation_id}, message_id={message_id}")
                return jsonify({"error": "Message not found", "status": "error"}), 404
            logger.info(f"Retrieved message | conversation_id={conversation_id}, message_id={message_id}")
            return jsonify(result.to_dict()), 200
        except Exception as e:
            logger.error(f"Error fetching message {message_id}: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    if request.method == "PATCH":
        data = request.get_json()
        if not data or "content" not in data:
            return jsonify({"error": "Missing 'content' field"}), 400
        try:
            logger.debug(f"Updating message | conversation_id={conversation_id}, message_id={message_id}")
            result = ConversationService.update_message(conversation_id, message_id, data["content"])
            if result is None:
                logger.warning(f"Message not found for update | message_id={message_id}")
                return jsonify({"error": "Message not found", "status": "error"}), 404
            logger.info(f"Updated message | conversation_id={conversation_id}, message_id={message_id}")
            return jsonify(result.to_dict()), 200
        except Exception as e:
            logger.error(f"Error updating message {message_id}: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    # DELETE
    try:
        logger.debug(f"Deleting message | conversation_id={conversation_id}, message_id={message_id}")
        deleted = ConversationService.delete_message(conversation_id, message_id)
        if not deleted:
            logger.warning(f"Message not found for delete | message_id={message_id}")
            return jsonify({"error": "Message not found", "status": "error"}), 404
        logger.info(f"Deleted message | conversation_id={conversation_id}, message_id={message_id}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500
