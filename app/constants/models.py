"""Model configuration constants."""

# LLM Models
LLM_MAIN_MODEL = "qwen2.5:3b"  # Used for RAG service — must support tool calling
LLM_LIGHT_MODEL = "qwen2.5:3b"  # Used for search and task breakdown

# Embedding Model
EMBEDDING_MODEL = "mxbai-embed-large"
