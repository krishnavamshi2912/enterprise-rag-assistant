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

def get_guard_llm() -> ChatGroq:
    """Initialize and return the configured guardrail LLM."""
    try:
        logger.info("Initializing guardrail LLM: %s", setting.GUARD_MODEL)
        llm = ChatGroq(model=setting.GUARD_MODEL,temperature=0,model_kwargs={"response_format": {"type": "json_object"}},)
        logger.info("Guardrail LLM initialized successfully")
        return llm

    except Exception:
        logger.exception("Failed to initialize guardrail LLM")
        raise