"""Provides the LLM used by the RAG application."""

from langchain_openai import ChatOpenAI
from app.gateway.gateway import get_gateway_llm, get_gateway_guardrail_llm
from app.logger import get_logger

logger = get_logger(__name__)

def get_llm() -> ChatOpenAI:
    """Return the main LLM through the Portkey gateway."""
    logger.info("Initializing main LLM through Portkey")
    return get_gateway_llm()

def get_guard_llm() -> ChatOpenAI:
    """Return the guardrail LLM through the Portkey gateway."""
    logger.info("Initializing guardrail LLM through Portkey")
    return get_gateway_guardrail_llm()