from flask import Flask
from flask_cors import CORS
from app.config.database import init_db
from app.services.rag_service import RAGService
from app.services.search_service import SearchService
from app.services.task_breakdown_service import TaskBreakdownService


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    init_db()
    app.task_breakdown_service = TaskBreakdownService()  # breaks down user requests into tasks
    app.rag_service = RAGService(task_breakdown_service=app.task_breakdown_service)  # loads knowledge from DB
    app.search_service = SearchService()  # web search service

    from app.controllers.conversation_controller import conversation_bp
    app.register_blueprint(conversation_bp)

    return app
