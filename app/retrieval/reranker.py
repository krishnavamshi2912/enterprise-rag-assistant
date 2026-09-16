"""Reranks retrieved documents using FlashRank."""

import time
from flashrank import Ranker, RerankRequest
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

_ranker = None


def get_reranker():
    """Initialize and return the FlashRank reranker."""
    global _ranker

    if _ranker is None:
        logger.info("Initializing FlashRank reranker | model=%s", setting.RERANKER_MODEL)
        try:
            _ranker = Ranker(model_name=setting.RERANKER_MODEL)
            logger.info("FlashRank reranker initialized successfully")
        except Exception:
            logger.exception("Failed to initialize FlashRank reranker")
            raise

    return _ranker


def rerank_documents(query: str, documents: list, top_k: int = 4) -> list:
    """Rerank retrieved LangChain documents and return the most relevant ones."""
    if not documents:
        logger.info("No documents available for reranking")
        return []

    start_time = time.time()

    try:
        ranker = get_reranker()

        passages = [
            {
                "id": index,
                "text": document.page_content,
            }
            for index, document in enumerate(documents)
        ]

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        results = ranker.rerank(request)

        logger.info(
            "FlashRank completed | candidates=%d | results=%d",
            len(documents),
            len(results),
        )

        for index, result in enumerate(results, start=1):
            logger.info(
                "Reranked candidate %d | score=%.4f | content=%s",
                index,
                result["score"],
                result["text"][:300].replace("\n", " "),
            )

        reranked_documents = []

        for result in results[:top_k]:
            document_index = result["id"]
            reranked_documents.append(documents[document_index])

        duration = time.time() - start_time

        logger.info(
            "Documents reranked | candidates=%d | selected=%d | duration=%.2fs",
            len(documents),
            len(reranked_documents),
            duration,
        )

        return reranked_documents

    except Exception:
        logger.exception("FlashRank reranking failed | returning original order")
        return documents[:top_k]