## Using this python file in order to create embeddings 


from app.configuration import setting
from langchain_community.embeddings import JinaEmbeddings

def get_embeddings_model():
    """Create a Jina embeddings model using the JINA_API_KEY environment variable."""
    return JinaEmbeddings(model_name=setting.JINA_EMBEDDING_MODEL)