"""Flask app factory for app2."""
from flask import Flask
from rag_application import RAGApplication


def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__)

    # Initialize RAG application
    app.rag_app = RAGApplication()

    # Register blueprints
    from rag_controller import rag_bp
    app.register_blueprint(rag_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8080)
