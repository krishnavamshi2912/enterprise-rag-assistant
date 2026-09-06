"""Provides the embedding model used by the RAG pipeline."""

from langchain_community.embeddings import JinaEmbeddings
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

def get_embeddings_model():
    """Initialize and return the configured Jina embedding model."""
    try:
        logger.info("Initializing embedding model: %s", setting.JINA_EMBEDDING_MODEL)

        embeddings = JinaEmbeddings(model_name=setting.JINA_EMBEDDING_MODEL)

        logger.info("Embedding model initialized successfully")
        return embeddings

    except Exception:
        logger.exception("Failed to initialize embedding model")
        raise