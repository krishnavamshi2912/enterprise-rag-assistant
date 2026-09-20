from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    route: str
    context: str
    answer: str
    retrieval_query: str
    blocked: bool