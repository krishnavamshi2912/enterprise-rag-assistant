## Using this python file in order to load the text file from Documents folder 

from app.configuration import setting
from langchain_community.document_loaders import TextLoader

def load_file(file_path: str = setting.DOCUMENTS_LOCATION):
    """Load a text file into LangChain Document objects."""
    try:
        text_loader = TextLoader(file_path, encoding="utf-8")
        documents = text_loader.load()
        return documents 
    except Exception as e:
            raise 
