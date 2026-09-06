### Using this python file in order to store the embeddings of the chunks in vector store 


import os 
from app.configuration import setting
from app.retrieval.embeddings import get_embeddings_model
from langchain_community.vectorstores import FAISS


def vector_store_build(chunks):
    """Embed document chunks and upload them to a vector store collection."""
    embeddings = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings)

def save_vector_store(vector_store, path: str = setting.VECTOR_STORE_PATH) -> None:
    """Saving the vector store locally"""
    vector_store.save_local(path)


def load_vector_store(path: str = setting.VECTOR_STORE_PATH):
    """Load the vector store details from local"""
    embeddings = get_embeddings_model()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def vector_store_exists(path: str = setting.VECTOR_STORE_PATH) -> bool:
    """Checking whether vector store exists"""
    return os.path.exists(os.path.join(path, "index.faiss"))

def get_retriver(vector_store, k:int = setting.TOP_K):
    """Get the top k results of the query"""
    return vector_store.as_retriever(search_kwargs={'k':k})