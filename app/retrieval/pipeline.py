"""Coordinates document ingestion, vector store creation, retrieval, and agent setup."""

from app.configuration import setting
from app.retrieval.agent import create_teleco_agent
from app.ingestion.loader import load_file
from app.ingestion.chunking import chunk_documents
from app.retrieval.llm import get_llm
from app.retrieval.tools import create_search_tool
from app.retrieval.vector_store import (
    vector_store_build,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriver
)
from app.logger import get_logger

logger = get_logger(__name__)

def build_vector_store_for_document(file_path: str = setting.DOCUMENTS_LOCATION):
    """Load an existing vector store or create one from the supplied document."""
    try:
        if vector_store_exists():
            logger.info("Found existing vector store | loading from disk")
            return load_vector_store()

        logger.info("No existing vector store found | starting vector store creation")

        documents = load_file(file_path)
        chunks = chunk_documents(documents)

        logger.info("Document processing completed | file=%s | chunks=%d",file_path,len(chunks))

        vector_store = vector_store_build(chunks)
        save_vector_store(vector_store)

        logger.info("Vector store created and saved successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to build vector store for document: %s", file_path)
        raise

def build_teleco_assistant(file_path: str = setting.DOCUMENTS_LOCATION):
    """Build the telecom assistant by connecting the vector store, retriever, search tool, and LLM."""
    try:
        logger.info("Building telecom assistant")

        vector_store = build_vector_store_for_document(file_path)
        retriver = get_retriver(vector_store)
        search_tool = create_search_tool(retriver)
        llm = get_llm()
        agent = create_teleco_agent(llm, [search_tool])

        logger.info("Telecom assistant created successfully")
        return agent
    except Exception:
        logger.exception("Failed to build telecom assistant")
        raise

def ask(agent, question: str) -> str:
    """Send a user question to the telecom assistant and return its final response."""
    try:
        logger.info("Processing user question: %s", question)

        response = agent.invoke(
            {'messages': [{'role': 'user', 'content': question}]}
        )

        answer = response['messages'][-1].content
        logger.info("User question processed successfully")

        return answer
    except Exception:
        logger.exception("Failed to process user question")
        raise