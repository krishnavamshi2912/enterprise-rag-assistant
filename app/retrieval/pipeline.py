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

def build_vector_store_for_document(file_path: str = setting.DOCUMENTS_LOCATION):
    if vector_store_exists():
        print("Found a saved vector store on disk, loading it....")
        return load_vector_store()
    print("No vector store found, building now...")
    documents = load_file(file_path)
    chunks = chunk_documents(documents)
    print(f"Loaded '{file_path}' and got {len(chunks)} chunks in total")

    vector_store = vector_store_build(chunks)
    save_vector_store(vector_store)
    print("Vector store saved in Documents folder")
    return vector_store 

def build_teleco_assistant(file_path: str = setting.DOCUMENTS_LOCATION):
    vector_store = build_vector_store_for_document(file_path)
    retriver = get_retriver(vector_store)
    search_tool = create_search_tool(retriver)
    llm = get_llm()
    agent = create_teleco_agent(llm,[search_tool])
    return agent

def ask(agent, question: str) -> str:
    response = agent.invoke({'messages':[{'role':'user', 'content': question}]})
    return response['messages'][-1].content
