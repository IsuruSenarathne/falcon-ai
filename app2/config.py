"""Configuration and environment variables."""
import os

# LLM Config
LLM_MODEL = "qwen2.5-coder:1.5b"
LLM_TEMPERATURE = 0.7

# Embeddings Config
EMBEDDINGS_MODEL = "mxbai-embed-large"

# Vector Store Config
VECTOR_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "pizza_reviews"
RETRIEVAL_K = 5

# Data Config
CSV_FILE_PATH = "/Users/isurusenarathne/Documents/Dev/AINemo/app2/pizza_store_reviews.csv"

# Prompt Template
SYSTEM_PROMPT = """
you are expert in restaurant reviews.

Here are some reviews: {reviews}
Here is the question: {question}

Provide your answer according to following rules:
1. use response as {{ answer: "your answer", reasoning: "your reasoning" }}
2. your answer should have more details and be more comprehensive.
3. your reasoning should be detailed and explain how you arrived at the answer.
"""
