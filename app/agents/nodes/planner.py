from app.agents.state import AgentState
from app.logger import get_logger
from app.retrieval.llm import get_llm

logger = get_logger(__name__)

llm = get_llm()

SIMPLE_RESPONSES = {
    "greeting": "Hello! How can I help you with Telecom BSS?",
    "thanks": "You're welcome!",
    "goodbye": "Goodbye!",
}

GREETING_WORDS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

THANKS_PHRASES = {
    "thanks",
    "thank you",
}

GOODBYE_PHRASES = {
    "bye",
    "goodbye",
}

HISTORY_KEYWORDS = [
    "last question",
    "previous question",
    "last message",
    "previous message",
    "what did i ask before",
    "what did i ask earlier",
    "last two questions",
    "last three questions",
    "previous two questions",
    "previous three questions",
]

def _is_greeting(message: str) -> bool:
    """Check whether the message is a greeting, optionally followed by a name."""
    words = message.split()

    if not words:
        return False

    if words[0] in {"hi", "hello", "hey"}:
        return len(words) <= 4

    if len(words) >= 2 and " ".join(words[:2]) in GREETING_WORDS:
        return len(words) <= 5

    return False

def _is_thanks(message: str) -> bool:
    """Check whether the message is a thank-you message."""
    return any(
        message.startswith(phrase)
        for phrase in THANKS_PHRASES
    )

def _is_goodbye(message: str) -> bool:
    """Check whether the message is a goodbye message."""
    return any(
        message.startswith(phrase)
        for phrase in GOODBYE_PHRASES
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
    """Analyze the query and decide whether retrieval is required."""
    messages = state["messages"]
    user_message = messages[-1]["content"].strip()
    normalized_message = user_message.lower()
    normalized_message = " ".join(normalized_message.split())

    if _is_greeting(normalized_message):
        logger.info("Simple greeting handled directly | question=%s", user_message)
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["greeting"]
        }

    if _is_thanks(normalized_message):
        logger.info("Simple thanks handled directly | question=%s", user_message)
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["thanks"]
        }

    if _is_goodbye(normalized_message):
        logger.info("Simple goodbye handled directly | question=%s", user_message)
        return {
            "route": "direct",
            "answer": SIMPLE_RESPONSES["goodbye"]
        }

    if any(keyword in normalized_message for keyword in HISTORY_KEYWORDS):
        answer = _get_history_answer(messages, normalized_message)

        logger.info("History query handled directly | question=%s", user_message)

        return {
            "route": "direct",
            "answer": answer
        }

    history = ""

    for message in messages[:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"

    prompt = f"""
You are an intelligent Planner for a Telecom BSS assistant.

Analyze the conversation history and the latest user message.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
"{user_message}"

Decide whether the latest message can be answered using only the conversation
history or whether Telecom BSS knowledge retrieval is required.

Return ONLY one of these two values:

CONVERSATIONAL
TECHNICAL

Use CONVERSATIONAL for:
- Casual conversation
- Questions that can be answered using the conversation history

Use TECHNICAL for:
- Telecom BSS questions
- Questions about rating, charging, billing, mediation, products,
  subscriptions, invoices, payments, revenue management, etc.
- Questions requiring information from the Telecom BSS knowledge base
"""

    response = llm.invoke(prompt)
    route = response.content.strip().upper()

    if route not in {"CONVERSATIONAL", "TECHNICAL"}:
        route = "TECHNICAL"

    logger.info("Planner route=%s | question=%s", route, user_message)

    return {
        "route": route.lower()
    }