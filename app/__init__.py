from flask import Flask
from flask_cors import CORS
from app.config.database import init_db
from app.services.rag_service import RAGService


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    init_db()
    app.rag_service = RAGService()

    from app.controllers.conversation_controller import conversation_bp
    app.register_blueprint(conversation_bp)

    return app
