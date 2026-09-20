from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriver import retriever_node
from app.agents.nodes.responder import responder_node
from app.guardrails.guardrails import input_guardrail_node, output_guardrail_node


def domain_rejection_node(state: AgentState) -> dict:
    """Return a fixed response for questions outside the supported domain."""
    return {
        "answer": state["answer"]
    }


def route_after_input_guardrail(state: AgentState) -> str:
    """Route blocked requests to END and safe requests to the planner."""
    if state["blocked"]:
        return "blocked"
    return "allowed"


def route_after_output_guardrail(state: AgentState) -> str:
    """Route blocked responses to END and safe responses to END."""
    if state["blocked"]:
        return "blocked"
    return "allowed"


def create_graph(retriever):
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node(
        "retriever",
        lambda state: retriever_node(state, retriever)
    )
    workflow.add_node("responder", responder_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("domain_rejection", domain_rejection_node)

    workflow.add_edge(START, "input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "blocked": END,
            "allowed": "planner",
        },
    )

    workflow.add_conditional_edges(
        "planner",
        lambda state: state["route"],
        {
            "direct": END,
            "technical": "retriever",
            "out_of_domain": "domain_rejection",
        },
    )

    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", "output_guardrail")

    workflow.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {
            "blocked": END,
            "allowed": END,
        },
    )

    workflow.add_edge("domain_rejection", END)

    return workflow.compile()