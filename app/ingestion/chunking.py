"""Splits Telecom BSS documents into clean, section-aware overlapping chunks."""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.configuration import setting
from app.logger import get_logger

logger = get_logger(__name__)

SECTION_PATTERN = r"(?m)(?=^\d+\.\s+[A-Z][^\n]*$)"
MIN_SECTION_CHUNK_SIZE = 200


def chunk_documents(documents):
    """Split documents by numbered sections while preserving section context."""
    try:
        logger.info(
            "Starting section-aware chunking | documents=%d | chunk_size=%d | chunk_overlap=%d",
            len(documents),
            setting.CHUNK_SIZE,
            setting.CHUNK_OVERLAP,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=setting.CHUNK_SIZE,
            chunk_overlap=setting.CHUNK_OVERLAP,
        )

        chunked_documents = []

        for document in documents:
            sections = re.split(SECTION_PATTERN, document.page_content)
            sections = [section.strip() for section in sections if section.strip()]

            for section in sections:
                lines = section.splitlines()
                section_name = lines[0].strip() if lines else "Unknown"

                section_document = document.model_copy(
                    update={
                        "page_content": section,
                        "metadata": {
                            **document.metadata,
                            "section": section_name,
                        },
                    }
                )

                section_chunks = text_splitter.split_documents(
                    [section_document]
                )

                for chunk in section_chunks:
                    if len(chunk.page_content.strip()) < MIN_SECTION_CHUNK_SIZE:
                        if chunked_documents:
                            previous_chunk = chunked_documents[-1]

                            if (
                                previous_chunk.metadata.get("section")
                                == section_name
                            ):
                                previous_chunk.page_content = (
                                    previous_chunk.page_content
                                    + "\n\n"
                                    + chunk.page_content
                                )
                                continue

                    chunked_documents.append(chunk)

        logger.info(
            "Section-aware chunking completed | chunks=%d",
            len(chunked_documents),
        )

        for index, chunk in enumerate(chunked_documents, start=1):
            content = chunk.page_content.strip()
            section = chunk.metadata.get("section", "unknown")

            logger.info(
                "Chunk %d | section=%s | characters=%d | preview=%s",
                index,
                section,
                len(content),
                content[:120].replace("\n", " "),
            )

        return chunked_documents

    except Exception:
        logger.exception("Section-aware document chunking failed")
        raise
