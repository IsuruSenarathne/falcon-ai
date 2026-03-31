import uuid

from flask import Blueprint, current_app, jsonify, request

from app.dto.conversation_dto import QueryRequest, SearchRequest
from app.services.conversation_service import ConversationService

conversation_bp = Blueprint("conversations", __name__)


@conversation_bp.route("/", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "AI Nemo"})


@conversation_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data["question"]
        # Check if query contains "search" keyword
        if "search" in question.lower():
            req = SearchRequest.from_json(data)
            result = current_app.search_service.search(req)
        else:
            req = QueryRequest.from_json(data)
            result = current_app.rag_service.query(req)

        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations", methods=["POST"])
def create_conversation():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data["question"]
        session_data = {**data, "session_id": str(uuid.uuid4())}

        # Check if query contains "search" keyword
        if "search" in question.lower():
            req = SearchRequest.from_json(session_data)
            result = current_app.search_service.search(req)
        else:
            req = QueryRequest.from_json(session_data)
            result = current_app.rag_service.query(req)

        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations", methods=["GET"])
def get_conversations():
    user_id = request.args.get("user_id")
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)

    try:
        result = ConversationService.get_conversations(user_id=user_id, limit=limit, offset=offset)
        return jsonify(result.to_dict()), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>", methods=["GET", "PATCH", "DELETE"])
def conversation(conversation_id: str):
    if request.method == "GET":
        try:
            result = ConversationService.get_conversation(conversation_id)
            if result is None:
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            return jsonify(result.to_dict()), 200
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    if request.method == "PATCH":
        data = request.get_json() or {}
        if "title" not in data:
            return jsonify({"error": "Missing 'title' field"}), 400
        try:
            result = ConversationService.update_conversation(conversation_id, data["title"])
            if result is None:
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            return jsonify(result.to_dict()), 200
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    # DELETE
    try:
        deleted = ConversationService.delete_conversation(conversation_id)
        if not deleted:
            return jsonify({"error": "Conversation not found", "status": "error"}), 404
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>/messages", methods=["GET", "POST"])
def messages(conversation_id: str):
    if request.method == "GET":
        try:
            result = ConversationService.get_messages(conversation_id)
            if result is None:
                return jsonify({"error": "Conversation not found", "status": "error"}), 404
            return jsonify(result.to_dict()), 200
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    # POST — append a new message to an existing conversation
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data["question"]
        session_data = {**data, "session_id": conversation_id}

        # Check if query contains "search" keyword
        if "search" in question.lower():
            req = SearchRequest.from_json(session_data)
            result = current_app.search_service.search(req)
        else:
            req = QueryRequest.from_json(session_data)
            result = current_app.rag_service.query(req)

        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@conversation_bp.route("/conversations/<conversation_id>/messages/<message_id>", methods=["GET", "PATCH", "DELETE"])
def message(conversation_id: str, message_id: str):
    if request.method == "GET":
        try:
            result = ConversationService.get_message(conversation_id, message_id)
            if result is None:
                return jsonify({"error": "Message not found", "status": "error"}), 404
            return jsonify(result.to_dict()), 200
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    if request.method == "PATCH":
        data = request.get_json()
        if not data or "content" not in data:
            return jsonify({"error": "Missing 'content' field"}), 400
        try:
            result = ConversationService.update_message(conversation_id, message_id, data["content"])
            if result is None:
                return jsonify({"error": "Message not found", "status": "error"}), 404
            return jsonify(result.to_dict()), 200
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    # DELETE
    try:
        deleted = ConversationService.delete_message(conversation_id, message_id)
        if not deleted:
            return jsonify({"error": "Message not found", "status": "error"}), 404
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
