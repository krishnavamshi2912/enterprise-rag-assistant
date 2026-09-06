"""Splits documents into overlapping chunks for retrieval."""

from app.configuration import setting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger import get_logger

logger = get_logger(__name__)

def chunk_documents(documents):
    """Split documents into overlapping chunks for retrieval."""
    try:
        logger.info("Starting document chunking | documents=%d | chunk_size=%d | chunk_overlap=%d",len(documents),setting.CHUNK_SIZE,setting.CHUNK_OVERLAP)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=setting.CHUNK_SIZE,
            chunk_overlap=setting.CHUNK_OVERLAP
        )

        chunked_documents = text_splitter.split_documents(documents)

        logger.info("Document chunking completed | chunks=%d",len(chunked_documents))
        return chunked_documents
        
    except Exception:
        logger.exception("Document chunking failed")
        raise

