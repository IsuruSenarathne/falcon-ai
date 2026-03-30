from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_service import RAGService
from database import init_db

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
    
    Request payload:
    {
        "question": "Your question here"
    }
    
    Response:
    {
        "question": "Your question here",
        "answer": "Generated answer from RAG chain",
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field in request"}), 400
        
        question = data["question"]
        
        # Query the RAG service
        answer = rag_service.query(question)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        }), 200
    
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
    
    Request payload:
    {
        "questions": [
            "Question 1",
            "Question 2"
        ]
    }
    
    Response:
    {
        "results": [
            {"question": "Question 1", "answer": "Answer 1"},
            {"question": "Question 2", "answer": "Answer 2"}
        ],
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or "questions" not in data:
            return jsonify({"error": "Missing 'questions' field in request"}), 400
        
        questions = data["questions"]
        
        # Process questions using the RAG service
        results = rag_service.batch_query(questions)
        
        return jsonify({
            "results": results,
            "status": "success"
        }), 200
    
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


if __name__ == "__main__":
    # Run Flask app
    # Set debug=True for development, debug=False for production
    app.run(debug=True, host="0.0.0.0", port=8080)
