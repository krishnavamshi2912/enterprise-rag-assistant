from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriver import retriever_node
from app.agents.nodes.responder import responder_node

def create_graph(retriever):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", lambda state: retriever_node(state, retriever))
    workflow.add_node("responder", responder_node)

    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges(
        "planner",
        lambda state: state["route"],
        {
            "direct": END,
            "conversational": "responder",
            "technical": "retriever",
        },
    )

    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()