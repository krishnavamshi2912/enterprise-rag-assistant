"""Provides a search tool for querying the telecom BSS knowledge base."""

from langchain.tools import tool
from app.logger import get_logger

logger = get_logger(__name__)

def create_search_tool(retriever):
    """Create a tool for searching the telecom BSS knowledge base."""

    @tool
    def search_telecom_knowledge(question: str) -> str:
        """Search the telecom BSS knowledge base for relevant information."""
        try:
            logger.info("Searching telecom knowledge base: %s", question)
            matching_chunks = retriever.invoke(question)
            logger.info("Knowledge search completed | chunks=%d",len(matching_chunks),)
            return "\n\n".join(chunk.page_content for chunk in matching_chunks)

        except Exception:
            logger.exception("Knowledge base search failed")
            raise

    return search_telecom_knowledge


