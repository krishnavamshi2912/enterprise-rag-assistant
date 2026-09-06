"""Builds, saves, loads, and retrieves documents from the FAISS vector store."""

import os

from app.configuration import setting
from app.retrieval.embeddings import get_embeddings_model
from langchain_community.vectorstores import FAISS
from app.logger import get_logger

logger = get_logger(__name__)

def vector_store_build(chunks):
    """Create a FAISS vector store by embedding the provided document chunks."""
    try:
        logger.info("Building vector store | chunks=%d", len(chunks))
        embeddings = get_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store built successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to build vector store")
        raise

def save_vector_store(vector_store, path: str = setting.VECTOR_STORE_PATH) -> None:
    """Save the FAISS vector store to the specified local path."""
    try:
        logger.info("Saving vector store | path=%s", path)
        vector_store.save_local(path)
        logger.info("Vector store saved successfully")
    except Exception:
        logger.exception("Failed to save vector store | path=%s", path)
        raise

def load_vector_store(path: str = setting.VECTOR_STORE_PATH):
    """Load the FAISS vector store from the specified local path."""
    try:
        logger.info("Loading vector store | path=%s", path)
        embeddings = get_embeddings_model()
        vector_store = FAISS.load_local(path,embeddings,allow_dangerous_deserialization=True,)
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to load vector store | path=%s", path)
        raise

def vector_store_exists(path: str = setting.VECTOR_STORE_PATH) -> bool:
    """Check whether the FAISS vector store exists at the specified path."""
    exists = os.path.exists(os.path.join(path, "index.faiss"))
    logger.info("Vector store existence check | path=%s | exists=%s", path, exists)
    return exists

def get_retriver(vector_store, k: int = setting.TOP_K):
    """Create a retriever that returns the top-k relevant document chunks."""
    logger.info("Creating retriever | top_k=%d", k)
    retriver = vector_store.as_retriever(search_kwargs={'k': k})
    logger.info("Retriever created successfully")
    return retriver
