from flask import Flask
from flask_cors import CORS
from app.config.database import init_db
from app.services.rag_service import RAGService
from app.services.search_service import SearchService
from app.services.task_breakdown_service import TaskBreakdownService
from app.services.thinking_service import ThinkingService


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    init_db()
    app.task_breakdown_service = TaskBreakdownService()  # breaks down user requests into tasks
    app.thinking_service = ThinkingService()  # decides which context source to use
    app.rag_service = RAGService(thinking_service=app.thinking_service)  # loads knowledge from DB
    app.search_service = SearchService(task_breakdown_service=app.task_breakdown_service)  # web search service

    from app.controllers.conversation_controller import conversation_bp
    app.register_blueprint(conversation_bp)

    return app
