"""Central configuration for environment variables and RAG application settings."""

import os
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

def get_required_env(name: str) -> str:
    """Retrieve a required environment variable and fail if it is unavailable."""
    value = os.getenv(name)
    if not value:
        logger.error("Required environment variable is missing: %s", name)
        raise ValueError(f"{name} is not set in the environment variables.")
    return value

class Settings:
    """Store configuration values used across the Telecom BSS RAG application."""

    # Groq LLM configuration and models
    # GROQ_API_KEY = get_required_env("GROQ_API_KEY")               ## Migrated the local LLM configuration to Portkey Gateway configuration 
    LLM_GROQ_MODEL = "openai/gpt-oss-120b"

    # Portkey LLM configuration 
    PORTKEY_API_KEY = get_required_env("PORTKEY_API_KEY")
    PORTKEY_PROVIDER = get_required_env("PORTKEY_PROVIDER")
    
    # Guard Model 
    GUARD_MODEL = "openai/gpt-oss-safeguard-20b"

    # JINA Embeddings and models
    JINA_API_KEY = get_required_env("JINA_API_KEY")
    JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"

    # Tracing 
    LANGSMITH_TRACING = get_required_env("LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT = get_required_env("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY = get_required_env("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = get_required_env("LANGSMITH_PROJECT")

    # Documents location
    DOCUMENTS_LOCATION = os.path.join("Documents", "Telecom_BSS_Knowledge_Base.txt")

    # Vector store
    #VECTOR_STORE_PATH = os.path.join("Documents", "faiss_index")
    QDRANT_URL = get_required_env("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = get_required_env("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = get_required_env("QDRANT_COLLECTION_NAME")

    # Chunking size and overlap config
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    # Retrieval results
    TOP_K = 8
    RERANK_TOP_K = 4
    RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"

setting = Settings()
logger.info("Application configuration loaded successfully")

