## Using this python file in order to chunk/split the data

from app.configuration import setting
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Split documents into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = setting.CHUNK_SIZE,
        chunk_overlap = setting.CHUNK_OVERLAP
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents