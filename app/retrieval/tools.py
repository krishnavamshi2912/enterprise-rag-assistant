from langchain.tools import tool 

def create_search_tool(retriver):
    """ Return the tool function that searches the telecom bss knowldge base"""

    @tool 
    def search_telecom_knowledge(question: str)-> str:
        """Search the telecom BSS knowledge base for relevant information."""
        matching_chunks = retriver.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)
    return search_telecom_knowledge
