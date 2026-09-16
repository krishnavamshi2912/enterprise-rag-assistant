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

    if route == "conversational":
        prompt = f"""
        You are a helpful Telecom BSS assistant.

        Answer the user's current message using the conversation history.

        CONVERSATION HISTORY:
        {history}

        CURRENT USER MESSAGE:
        {question}

        If the user asks about a previous question or message, use the conversation
        history to answer it directly.

        Respond naturally and concisely.
        """
    else:
        prompt = f"""
        You are a Telecom BSS assistant.

        Answer the user's question using the provided Telecom BSS context.

        CONVERSATION HISTORY:
        {history}

        TELECOM BSS CONTEXT:
        {context}

        CURRENT USER QUESTION:
        {question}

        Use the provided context to answer the question.

        If the context does not contain enough information, clearly say that you
        do not have enough information.
        """

    response = llm.invoke(prompt)

    logger.info("Response generated | route=%s", route)

    return {"answer": response.content}