from flask import Flask
from flask_cors import CORS
from langchain_core.documents import Document
from app.config.database import init_db
from app.services.agent_service import AgentService
from app.services.vector_store_service import VectorStoreService
from app.services.search_service import SearchService
from app.repositories.knowledge_repository import KnowledgeRepository
from app.tools import init_tools
from app.constants.models import EMBEDDING_MODEL


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    init_db()

    # Load knowledge documents into vector store
    documents = KnowledgeRepository.load_documents()
    docs = [Document(page_content=doc) for doc in documents]

    vector_store = VectorStoreService(
        db_path="chroma_db",
        collection_name="knowledge_base",
        embeddings_model=EMBEDDING_MODEL,
        retrieval_k=5,
    )
    vector_store.add_documents(docs, [str(i) for i in range(len(docs))])

    # Inject service instances into tools before agent is created
    init_tools(search_service=SearchService(), vector_store=vector_store)

    app.agent_service = AgentService()

    from app.controllers.conversation_controller import conversation_bp
    from app.controllers.feedback_controller import feedback_bp

    app.register_blueprint(conversation_bp)
    app.register_blueprint(feedback_bp)

    return app
