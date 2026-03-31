import uuid

from flask import Blueprint, current_app, jsonify, request

from app.dto.conversation_dto import QueryRequest
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
        req = QueryRequest.from_json({**data, "session_id": str(uuid.uuid4())})
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
        req = QueryRequest.from_json({**data, "session_id": conversation_id})
        result = current_app.rag_service.query(req)
        status_code = 200 if result.status == "success" else 400
        return jsonify(result.to_dict()), status_code
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
