from app.logger import get_logger

logger = get_logger(__name__)

def retriever_node(state: dict, retriever) -> dict:
    """Retrieve relevant Telecom BSS documents for the current user question."""
    question = state["messages"][-1]["content"]

    logger.info("Retrieving context | question=%s", question)

    documents = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in documents)

    logger.info("Retrieved documents=%d", len(documents))

    return {"context": context}