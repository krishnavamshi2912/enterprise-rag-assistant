from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    route: str
    context: str
    answer: str