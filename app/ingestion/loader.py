"""Loads text documents from the configured documents directory."""

from langchain_community.document_loaders import TextLoader
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

def load_file(file_path: str = setting.DOCUMENTS_LOCATION):
    """Load a text file into LangChain Document objects."""
    try:
        logger.info("Loading document: %s", file_path)

        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        logger.info("Document loaded successfully: %s", file_path)
        logger.info("Loaded %d document(s)", len(documents))

        return documents

    except Exception as e:
        logger.error("Failed to load document: %s", file_path)
        logger.exception("Document loading error: %s", e)
        raise