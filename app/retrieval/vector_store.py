"""Builds, saves, loads, and retrieves documents from the FAISS vector store."""

import os

from app.configuration import setting
from app.retrieval.embeddings import get_embeddings_model
from langchain_qdrant import QdrantVectorStore       # Adding dependent Qdrant imports
from qdrant_client import QdrantClient
from app.logger import get_logger

logger = get_logger(__name__)

def vector_store_build(chunks):
    """Create a QDRANT vector store by embedding the provided document chunks."""
    try:
        logger.info("Building vector store | chunks=%d", len(chunks))
        embeddings = get_embeddings_model()
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url = setting.QDRANT_URL,
            api_key = setting.QDRANT_API_KEY,
            collection_name = setting.QDRANT_COLLECTION_NAME
        )
        logger.info("Qdrant Vector store built successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to build vector store")
        raise

def load_vector_store():
    """Load the collection from Qdrant cloud"""
    try:
        logger.info("Connecting to Qdrant cloud")
        embeddings = get_embeddings_model()
        vector_store = QdrantVectorStore.from_existing_collection(
            setting.QDRANT_COLLECTION_NAME,
            embeddings,
            url = setting.QDRANT_URL,
            api_key = setting.QDRANT_API_KEY
        )
        logger.info("Qdrant Vector store loaded successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to load Qdrant vector store... ")
        raise

def vector_store_exists() -> bool:
    """Check whether the Qdrant vector store exists."""
    client = QdrantClient(
        url = setting.QDRANT_URL,
        api_key = setting.QDRANT_API_KEY,
    )
    logger.info("Verifying '%s' collection exists in Qdrant cloud",setting.QDRANT_COLLECTION_NAME)
    return client.collection_exists(setting.QDRANT_COLLECTION_NAME)

def get_retriver(vector_store, k: int = setting.TOP_K):
    """Create a retriever that returns the top-k relevant document chunks."""
    logger.info("Creating retriever | top_k=%d", k)
    retriver = vector_store.as_retriever(search_kwargs={'k': k})
    logger.info("Retriever created successfully")
    return retriver
