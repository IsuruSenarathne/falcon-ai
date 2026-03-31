from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_service import RAGService
from database import init_db, SessionLocal
from db_service import ConversationService

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Initialize RAG Service
rag_service = RAGService()

# Initialize Database (creates tables if they don't exist)
init_db()


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "RAG API"})


@app.route("/query", methods=["POST"])
def query():
    """
    Accept a question via POST request and return RAG response.
    Conversation is automatically stored in the database.
    
    Request payload:
    {
        "question": "Your question here",
        "user_id": "optional_user_id",
        "session_id": "optional_session_id"
    }
    
    Response:
    {
        "conversation_id": "uuid-string",
        "question": "Your question here",
        "answer": "Generated answer from RAG chain",
        "status": "success",
        "response_time": 1.23,
        "created_at": "2026-03-30T..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field in request"}), 400
        
        question = data["question"]
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        
        # Query the RAG service (stores conversation automatically)
        result = rag_service.query(question, user_id=user_id, session_id=session_id)
        
        return jsonify(result), 200 if result.get("status") == "success" else 400
    
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/batch-query", methods=["POST"])
def batch_query():
    """
    Accept multiple questions via POST request and return RAG responses.
    All conversations are automatically stored in the database.
    
    Request payload:
    {
        "questions": [
            "Question 1",
            "Question 2"
        ],
        "user_id": "optional_user_id",
        "session_id": "optional_session_id"
    }
    
    Response:
    {
        "results": [
            {
                "conversation_id": "uuid-1",
                "question": "Question 1",
                "answer": "Answer 1",
                "status": "success",
                "response_time": 1.23
            },
            {
                "conversation_id": "uuid-2",
                "question": "Question 2",
                "answer": "Answer 2",
                "status": "success",
                "response_time": 1.45
            }
        ],
        "session_id": "optional_session_id",
        "total": 2,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or "questions" not in data:
            return jsonify({"error": "Missing 'questions' field in request"}), 400
        
        questions = data["questions"]
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        
        # Process questions using the RAG service (stores all conversations automatically)
        result = rag_service.batch_query(questions, user_id=user_id, session_id=session_id)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/conversations", methods=["GET"])
def get_conversations():
    """
    Fetch all conversations from the database with optional filters.
    
    Query parameters:
    - user_id: Filter by user ID (optional)
    - session_id: Filter by session ID (optional)
    - limit: Maximum number of results to return (default: 100, max: 1000)
    - offset: Number of results to skip (default: 0)
    
    Response:
    {
        "conversations": [...],
        "total": 150,
        "limit": 100,
        "offset": 0,
        "status": "success"
    }
    """
    try:
        # Get query parameters
        user_id = request.args.get("user_id")
        session_id = request.args.get("session_id")
        limit = request.args.get("limit", default=100, type=int)
        offset = request.args.get("offset", default=0, type=int)
        
        # Fetch conversations from database service (handles all database logic)
        result = ConversationService.get_conversations_list(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/conversations/<conversation_id>/messages", methods=["GET", "POST"])
def get_messages(conversation_id):
    """
    Fetch all messages for a specific conversation, ordered chronologically.

    Response:
    {
        "conversation_id": "uuid",
        "messages": [
            {"message_id": "...", "role": "user", "content": "...", "created_at": "..."},
            {"message_id": "...", "role": "bot",  "content": "...", "created_at": "..."}
        ],
        "status": "success"
    }
    """
    if request.method == "GET":
        try:
            result = ConversationService.get_messages(conversation_id)
            status_code = 404 if result.get("status") == "error" else 200
            return jsonify(result), status_code
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    # POST — send a new message to an existing conversation
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field in request"}), 400

        question = data["question"]
        user_id = data.get("user_id")

        result = rag_service.query(question, user_id=user_id, session_id=conversation_id)
        return jsonify(result), 200 if result.get("status") == "success" else 400

    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    # Run Flask app
    # Set debug=True for development, debug=False for production
    app.run(debug=True, host="0.0.0.0", port=8080)
