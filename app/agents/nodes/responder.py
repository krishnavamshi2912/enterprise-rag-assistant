from app.agents.state import AgentState
from app.logger import get_logger
from app.retrieval.llm import get_llm

logger = get_logger(__name__)
llm = get_llm()


def responder_node(state: AgentState) -> dict:
    """Generate the final response using conversation history and retrieved context."""
    messages = state["messages"]
    route = state["route"]
    context = state.get("context", "")

    history = ""
    for message in messages[:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"

    question = messages[-1]["content"] if messages else ""

    prompt = f"""
You are a Telecom BSS knowledge assistant.

Answer the user's current question using ONLY the provided Telecom BSS context
and relevant conversation history.

CONVERSATION HISTORY:
{history}

TELECOM BSS CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

RULES:
1. Use the provided context as the source of truth.
2. Do not add facts, examples, algorithms, rules, or technical details that are
   not supported by the provided context.
3. You may use conversation history to understand follow-up questions, but do not
   use previous answers as a source of new factual information.
4. If the provided context does not contain enough information to answer the
   question, clearly say that the information is not available in the provided
   Telecom BSS context.
5. Do not guess or fill missing information using your general knowledge.
6. Answer directly and concisely.
7. Use simple Markdown with headings or bullet points when useful.
8. Avoid Markdown tables unless the user explicitly asks for a table.
"""

    response = llm.invoke(prompt)

    logger.info("Response generated | route=%s", route)

    return {"answer": response.content}