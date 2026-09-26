from typing import Literal
from pydantic import BaseModel, Field
from app.agents.state import AgentState
from app.logger import get_logger
from app.retrieval.llm import get_llm
from app.utils.query_classifier import (
    is_greeting,
    is_thanks,
    is_goodbye,
    is_history_query,
    normalize_message,
)

logger = get_logger(__name__)
llm = get_llm()


class PlannerDecision(BaseModel):
    """Structured output returned by the planner LLM."""
    route: Literal["TECHNICAL", "OUT_OF_DOMAIN"]
    retrieval_query: str = Field(
        description="Short retrieval query for semantic search."
    )


planner_llm = llm.with_structured_output(PlannerDecision)


SIMPLE_RESPONSES = {
    "greeting": "Hello! How can I help you with Telecom BSS?",
    "thanks": "You're welcome!",
    "goodbye": "Goodbye!",
}


DOMAIN_REJECTION_RESPONSE = (
    "I can help with questions related to Telecom BSS, such as "
    "rating, charging, billing, mediation, subscriptions, invoices, "
    "payments, and related concepts."
)


def _get_history_answer(messages: list, message: str) -> str:
    """Return an answer for questions about previous user messages."""
    previous_user_messages = [
        item["content"]
        for item in messages[:-1]
        if item["role"] == "user"
    ]

    if not previous_user_messages:
        return "I don't have any previous questions in this conversation."

    if "last two" in message or "previous two" in message:
        questions = previous_user_messages[-2:]
    elif "last three" in message or "previous three" in message:
        questions = previous_user_messages[-3:]
    else:
        questions = previous_user_messages[-1:]

    if len(questions) == 1:
        return f'Your last question was: "{questions[0]}"'

    return "Your previous questions were:\n" + "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(questions, start=1)
    )


def planner_node(state: AgentState) -> dict:
    """Classify the query and create a retrieval query for technical questions."""
    messages = state["messages"]
    user_message = messages[-1]["content"].strip()

    normalized_message = normalize_message(user_message)

    if is_greeting(normalized_message):
        logger.info(
            "Simple greeting handled directly | question=%s",
            user_message,
        )
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["greeting"],
            "retrieval_query": "",
        }

    if is_thanks(normalized_message):
        logger.info(
            "Simple thanks handled directly | question=%s",
            user_message,
        )
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["thanks"],
            "retrieval_query": "",
        }

    if is_goodbye(normalized_message):
        logger.info(
            "Simple goodbye handled directly | question=%s",
            user_message,
        )
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["goodbye"],
            "retrieval_query": "",
        }

    if is_history_query(normalized_message):
        answer = _get_history_answer(
            messages,
            normalized_message,
        )
        logger.info(
            "History query handled directly | question=%s",
            user_message,
        )
        return {
            "route": "direct",
            "answer": answer,
            "retrieval_query": "",
        }

    history = ""
    for message in messages[-4:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"

    prompt = f"""
    You are a planner for a Telecom BSS RAG assistant.

    Classify the latest user message as:
    - TECHNICAL: Telecom BSS question or relevant follow-up.
    - OUT_OF_DOMAIN: unrelated question.

    For technical questions, identify the main technical concept and use that as the retrieval query.
    Do not include generic domain words unless they are necessary to distinguish the concept.

    Rules:
    - Remove conversational filler.
    - Preserve the user's specific topic and intent.
    - Do not introduce new concepts.
    - Avoid repeating "Telecom BSS" when unnecessary.
    - Use conversation history to resolve follow-ups.
    - For OUT_OF_DOMAIN, retrieval_query must be empty.

    Conversation history:
    {history}

    User message:
    "{user_message}"
    """

    decision = planner_llm.invoke(prompt)

    route = decision.route
    retrieval_query = decision.retrieval_query.strip()

    if route == "OUT_OF_DOMAIN":
        logger.info(
            "Planner route=OUT_OF_DOMAIN | question=%s",
            user_message,
        )
        return {
            "route": "out_of_domain",
            "answer": DOMAIN_REJECTION_RESPONSE,
            "retrieval_query": "",
        }

    if not retrieval_query:
        retrieval_query = user_message

    logger.info(
        "Planner route=TECHNICAL | retrieval_query=%s | question=%s",
        retrieval_query,
        user_message,
    )

    return {
        "route": "technical",
        "retrieval_query": retrieval_query,
    }