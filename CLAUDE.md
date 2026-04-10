# AINemo App Structure

## Stack
Flask 3.0 · SQLAlchemy 2.0 · MySQL (PyMySQL) · LangChain 0.3 · ChromaDB · Ollama (llama3.2:1b + nomic-embed-text)

## Entry Points
- `main.py` → `create_app()` → runs on :8080
- `app/__init__.py` → factory: init CORS, DB, RAGService, register blueprint

## File Map
```
app/
  __init__.py              # app factory
  config/database.py       # SQLAlchemy engine, SessionLocal, Base, init_db()
  models/conversation.py   # Conversation, Message ORM; MessageRole/Status enums
  controllers/
    conversation_controller.py  # Blueprint, all route handlers
  services/
    conversation_service.py     # save_exchange(), CRUD wrappers
    rag_service.py              # RAG pipeline: embed → retrieve → LLM → persist
  repositories/
    conversation_repository.py  # DB CRUD for Conversation/Message
    knowledge_repository.py     # Loads courses/advisors/depts/modules as text docs
  dto/conversation_dto.py       # QueryRequest, QueryResponse, ConversationDTO, MessageDTO
```

## DB Models
| Table | Key Cols |
|-------|----------|
| conversations | id, conversation_id(UUID), user_id, title, created_at, updated_at |
| messages | id, message_id(UUID), conversation_id(FK), role(USER\|BOT), content, status(SUCCESS\|ERROR\|PENDING), response_time |

## API Routes
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | health check |
| POST | `/query` | one-off Q&A (no session) |
| POST | `/conversations` | create conv + first message |
| GET | `/conversations` | list (limit, offset, user_id) |
| GET/PATCH/DELETE | `/conversations/:id` | get\|update title\|delete |
| GET/POST | `/conversations/:id/messages` | list\|add message |
| GET/PATCH/DELETE | `/conversations/:id/messages/:mid` | get\|update\|delete msg |

## RAG Flow
1. Startup: `KnowledgeRepository.load_documents()` → DB tables → plain-text docs → ChromaDB
2. Query: question → `OllamaEmbeddings` → vector search → context → `ChatOllama` → answer
3. Persist: `ConversationService.save_exchange()` → MySQL

## Env Vars
`DB_USER` `DB_PASSWORD` `DB_HOST`(localhost) `DB_PORT`(3306) `DB_NAME`(nemo)

## Conventions
- Services call repositories; controllers call services
- DTOs used for request/response shapes (dataclasses)
- Blueprint registered at `/` with no prefix
- HTML-formatted LLM responses (inner HTML only)
