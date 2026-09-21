import json
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

primary_llm_slug = setting.PORTKEY_PROVIDER

def get_gateway_llm() -> ChatOpenAI:
    """Return the main chat model routed through Portkey."""
    logger.info("Routing main LLM through Portkey | provider=%s | model=%s",primary_llm_slug,setting.LLM_GROQ_MODEL,)
    headers = createHeaders(api_key=setting.PORTKEY_API_KEY, provider=primary_llm_slug)
    return ChatOpenAI(
        api_key="portkey",  # dummy value - the real auth is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        model=setting.LLM_GROQ_MODEL,
        default_headers=headers,
    )

def get_gateway_guardrail_llm() -> ChatOpenAI:
    """Return the guardrail chat model routed through Portkey."""
    logger.info("Routing guardrail LLM through Portkey | provider=%s | model=%s",primary_llm_slug,setting.GUARD_MODEL,)
    headers = createHeaders(api_key=setting.PORTKEY_API_KEY, provider=primary_llm_slug)
    return ChatOpenAI(
        api_key="portkey",  # dummy value - the real auth is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        model=setting.GUARD_MODEL,
        default_headers=headers,
    )