"""Configuration and environment variables."""
import os

# LLM Config
LLM_MODEL = "qwen2.5-coder:1.5b"
LLM_TEMPERATURE = 0.7

# Embeddings Config
EMBEDDINGS_MODEL = "mxbai-embed-large"

# Vector Store Config
VECTOR_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "documents"
RETRIEVAL_K = 5

# Data Config
CSV_FILE_PATH = "/Users/isurusenarathne/Documents/Dev/AINemo/app2/pizza_store_reviews.csv"

# Prompt Template
SYSTEM_PROMPT = """
You are a helpful assistant answering questions based on provided context.

Context: {context}
Question: {question}

Provide your answer according to following rules:
1. Use response format as {{ answer: "your answer", reasoning: "your reasoning" }}
2. Your answer should be detailed and comprehensive.
3. Your reasoning should clearly explain how you arrived at the answer based on the context.
"""
