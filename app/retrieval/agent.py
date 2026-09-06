"""Creates the agent responsible for handling telecom BSS queries."""

from app.configuration import setting
from app.logger import get_logger
from langchain.agents import create_agent

logger = get_logger(__name__)

def create_teleco_agent(llm, tools):
    """Create the telecom BSS agent using the configured LLM and tools."""
    try:
        logger.info("Creating telecom agent | tools=%d", len(tools))
        agent = create_agent(model=llm,tools=tools,system_prompt=setting.SYSTEM_PROMPT,)
        logger.info("Telecom agent created successfully")
        return agent
    except Exception:
        logger.exception("Failed to create telecom agent")
        raise