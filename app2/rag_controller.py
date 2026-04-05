"""API controller for RAG application."""
import uuid
import logging
from flask import Blueprint, current_app, jsonify, request
from datetime import datetime

logger = logging.getLogger(__name__)

rag_bp = Blueprint("rag", __name__)

# In-memory storage for sessions (for app2 simplicity)
sessions = {}


@rag_bp.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    logger.info("GET / - Health check")
    return jsonify({
        "status": "healthy",
        "service": "AI Nemo App2",
        "version": "1.0"
    }), 200


@rag_bp.route("/query", methods=["POST"])
def query():
    """One-off query endpoint (no session)."""
    data = request.get_json()
    if not data or "question" not in data:
        logger.warning("❌ Missing 'question' field")
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data.get("question", "")
        logger.info(f"POST /query - Question: {question[:50]}...")

        if not question.strip():
            logger.warning("Empty question")
            return jsonify({"error": "Question cannot be empty"}), 400

        # Execute query
        response = current_app.rag_app.query(question)

        logger.info("Query completed")
        return jsonify({
            "status": "success",
            "question": question,
            "answer": response,
            "timestamp": datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@rag_bp.route("/sessions", methods=["POST"])
def create_session():
    """Create a new session/conversation."""
    data = request.get_json()
    if not data or "question" not in data:
        logger.warning("Missing 'question' field")
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data.get("question", "")
        session_id = str(uuid.uuid4())

        logger.info(f"POST /sessions - Creating session {session_id}")
        logger.info(f"First question: {question[:50]}...")

        if not question.strip():
            logger.warning("Empty question")
            return jsonify({"error": "Question cannot be empty"}), 400

        # Execute query
        response = current_app.rag_app.query(question)

        # Store session
        sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [
                {"role": "user", "content": question, "timestamp": datetime.utcnow().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.utcnow().isoformat()}
            ]
        }

        logger.info(f"Session created: {session_id}")
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "question": question,
            "answer": response,
            "created_at": sessions[session_id]["created_at"]
        }), 201

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@rag_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all sessions."""
    logger.info("GET /sessions - Listing sessions")
    try:
        session_list = [
            {
                "session_id": sid,
                "created_at": s["created_at"],
                "message_count": len(s["messages"])
            }
            for sid, s in sessions.items()
        ]

        logger.info(f"Found {len(session_list)} sessions")
        return jsonify({
            "status": "success",
            "sessions": session_list,
            "total": len(session_list)
        }), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@rag_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get a specific session."""
    logger.info(f"GET /sessions/{session_id}")
    try:
        if session_id not in sessions:
            logger.warning(f"Session not found: {session_id}")
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        logger.info(f"Retrieved session {session_id}")
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "created_at": session["created_at"],
            "messages": session["messages"]
        }), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@rag_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session."""
    logger.info(f"DELETE /sessions/{session_id}")
    try:
        if session_id not in sessions:
            logger.warning(f"Session not found: {session_id}")
            return jsonify({"error": "Session not found"}), 404

        del sessions[session_id]
        logger.info(f"Session deleted: {session_id}")
        return jsonify({
            "status": "success",
            "message": "Session deleted"
        }), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@rag_bp.route("/sessions/<session_id>/query", methods=["POST"])
def add_message(session_id: str):
    """Add a message to an existing session."""
    data = request.get_json()
    if not data or "question" not in data:
        logger.warning("Missing 'question' field")
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        question = data.get("question", "")
        logger.info(f"POST /sessions/{session_id}/query - Question: {question[:50]}...")

        if session_id not in sessions:
            logger.warning(f"Session not found: {session_id}")
            return jsonify({"error": "Session not found"}), 404

        if not question.strip():
            logger.warning("Empty question")
            return jsonify({"error": "Question cannot be empty"}), 400

        # Execute query
        response = current_app.rag_app.query(question)

        # Add to session
        sessions[session_id]["messages"].append({
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat()
        })
        sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.info(f"Message added to session {session_id}")
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "question": question,
            "answer": response,
            "message_count": len(sessions[session_id]["messages"])
        }), 200

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
