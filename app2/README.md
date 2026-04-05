# App2 - RAG Application

A simplified RAG (Retrieval-Augmented Generation) application with Flask REST API and optional web search integration.

## Quick Start

```bash
cd app2
pip install -r requirements.txt
python main.py
```

Server runs on `http://localhost:8080`

## API Endpoints

### Health Check
```bash
GET /
```
Response:
```json
{
  "status": "healthy",
  "service": "AI Nemo App2",
  "version": "1.0"
}
```

### One-Off Query
```bash
POST /query
Content-Type: application/json

{
  "question": "What are the best practices for RAG?"
}
```
Response:
```json
{
  "status": "success",
  "question": "What are the best practices for RAG?",
  "answer": "...",
  "timestamp": "2026-04-05T10:30:00"
}
```

### Create Session (Start Conversation)
```bash
POST /sessions
Content-Type: application/json

{
  "question": "Tell me about pizza"
}
```
Response:
```json
{
  "status": "success",
  "session_id": "uuid-here",
  "question": "Tell me about pizza",
  "answer": "...",
  "created_at": "2026-04-05T10:30:00"
}
```

### List Sessions
```bash
GET /sessions
```
Response:
```json
{
  "status": "success",
  "sessions": [
    {
      "session_id": "uuid-1",
      "created_at": "2026-04-05T10:30:00",
      "message_count": 2
    }
  ],
  "total": 1
}
```

### Get Session (View Conversation)
```bash
GET /sessions/{session_id}
```
Response:
```json
{
  "status": "success",
  "session_id": "uuid-here",
  "created_at": "2026-04-05T10:30:00",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ]
}
```

### Add Message to Session (Continue Conversation)
```bash
POST /sessions/{session_id}/query
Content-Type: application/json

{
  "question": "Follow-up question?"
}
```

### Delete Session
```bash
DELETE /sessions/{session_id}
```

## Web Search

To use web search in queries, include "websearch" in your question:

```bash
POST /query

{
  "question": "websearch latest trends in AI"
}
```

The app will automatically:
1. Detect the "websearch" keyword
2. Fetch results from Brave Search API
3. Generate response based on web content

**Required**: Set `BRAVE_API_KEY` in `.env` file

## Architecture

```
main.py                  # Flask entry point + .env loader
app.py                   # Flask app factory
rag_controller.py        # REST API endpoints
rag_application.py       # RAG orchestration
├── document_loader.py   # CSV → Documents
├── vector_store_service.py # ChromaDB management
├── llm_chain_service.py # LLM prompting
├── search_service.py    # Web search (Brave API)
├── search_retriever.py  # Web search retriever
└── response_formatter.py # Output formatting
config.py               # Configuration
```

## Configuration

Edit `config.py` to customize:
- LLM model
- Embedding model
- Vector DB path
- Retrieval k (number of docs to retrieve)
- CSV data source
- System prompt

## Logging

All operations are logged with structured logging. Check console for:
- Initialization steps
- Query routing (vector search vs web search)
- Retrieval progress
- LLM invocation details

## Features

✅ Local vector search (ChromaDB)
✅ Web search integration (Brave Search API)
✅ Session/Conversation management
✅ Flexible retriever system
✅ Comprehensive logging
✅ Generic templates (works with any data)
