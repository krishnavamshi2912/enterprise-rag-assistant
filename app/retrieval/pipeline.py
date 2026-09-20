"""Coordinates document ingestion, vector store creation, and LangGraph assistant setup."""

from app.configuration import setting
from app.ingestion.loader import load_file
from app.ingestion.chunking import chunk_documents
from app.retrieval.vector_store import (
    vector_store_build,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriver
)
from app.logger import get_logger
from app.retrieval.tracing import check_langsmith_tracing
from app.agents.graph import create_graph

logger = get_logger(__name__)

chat_history = []

def build_vector_store_for_document(file_path: str = setting.DOCUMENTS_LOCATION):
    """Load an existing vector store or create one from the supplied document."""
    try:
        if vector_store_exists():
            logger.info("Found existing vector store | loading from disk")
            return load_vector_store()

        logger.info("No existing vector store found | starting vector store creation")
        documents = load_file(file_path)
        chunks = chunk_documents(documents)

        logger.info(
            "Document processing completed | file=%s | chunks=%d",
            file_path,
            len(chunks)
        )

        vector_store = vector_store_build(chunks)
        save_vector_store(vector_store)

        logger.info("Vector store created and saved successfully")
        return vector_store

    except Exception:
        logger.exception("Failed to build vector store for document: %s", file_path)
        raise

def build_teleco_assistant(file_path: str = setting.DOCUMENTS_LOCATION):
    """Build the telecom assistant using the LangGraph workflow."""
    try:
        logger.info("Building telecom assistant")
        check_langsmith_tracing()

        vector_store = build_vector_store_for_document(file_path)
        retriver = get_retriver(vector_store)

        graph = create_graph(retriver)

        logger.info("Telecom assistant created successfully")
        return graph

    except Exception:
        logger.exception("Failed to build telecom assistant")
        raise

def ask(graph, question: str) -> str:
    """Send a user question through the LangGraph workflow."""
    global chat_history

    try:
        logger.info("Processing user question: %s", question)

        messages = chat_history + [
            {"role": "user", "content": question}
        ]

        response = graph.invoke({
            "messages": messages,
            "route": "",
            "retrieval_query": "",
            "context": "",
            "answer": "",
            "blocked": False
        })

        answer = response["answer"]

        if not response["blocked"]:
            chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

        logger.info("User question processed successfully")
        return answer

    except Exception:
        logger.exception("Failed to process user question")
        raise