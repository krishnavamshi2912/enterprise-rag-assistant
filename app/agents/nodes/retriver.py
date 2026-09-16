from app.configuration import setting
from app.logger import get_logger
from app.retrieval.reranker import rerank_documents

logger = get_logger(__name__)


def retriever_node(state: dict, retriever) -> dict:
    """Retrieve and rerank relevant Telecom BSS documents."""
    question = state["messages"][-1]["content"]

    logger.info("Retrieving context | question=%s", question)

    documents = retriever.invoke(question)

    logger.info("Retrieved documents=%d", len(documents))

    for index, document in enumerate(documents, start=1):
        content = document.page_content.strip()
        source = document.metadata.get("source", "unknown")

        logger.info(
            "Retrieved chunk %d | source=%s | characters=%d",
            index,
            source,
            len(content),
        )

    reranked_documents = rerank_documents(
        question,
        documents,
        top_k=setting.RERANK_TOP_K,
    )

    logger.info(
        "Reranking completed | candidates=%d | selected=%d",
        len(documents),
        len(reranked_documents),
    )

    for index, document in enumerate(reranked_documents, start=1):
        content = document.page_content.strip()

        # logger.info(
        #     "Selected chunk %d | characters=%d | content=%s",
        #     index,
        #     len(content),
        #     content[:300].replace("\n", " "),
        # )

        logger.info(
            "Selected chunk %d | characters=%d ",
            index,
            len(content),
        )

    context = "\n\n".join(
        document.page_content
        for document in reranked_documents
    )

    logger.info(
        "Retrieved context prepared | chunks=%d | characters=%d",
        len(reranked_documents),
        len(context),
    )

    return {"context": context}