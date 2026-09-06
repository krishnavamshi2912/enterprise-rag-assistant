"""Provides the LLM used by the RAG application."""

from langchain_groq import ChatGroq
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

def get_llm() -> ChatGroq:
    """Initialize and return the configured llm chat model."""
    try:
        logger.info("Initializing LLM: %s", setting.LLM_GROQ_MODEL)
        llm = ChatGroq(model=setting.LLM_GROQ_MODEL,temperature=0)
        logger.info("LLM initialized successfully")
        return llm
    
    except Exception:
        logger.exception("Failed to initialize LLM")
        raise