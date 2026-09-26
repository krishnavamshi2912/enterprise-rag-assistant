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


def normalize_message(message: str) -> str:
    """Normalize a user message for deterministic classification."""
    return " ".join(message.lower().split())


def is_greeting(message: str) -> bool:
    """Check whether the message is a simple greeting."""
    words = message.split()

    if not words:
        return False

    if words[0] in {"hi", "hello", "hey"}:
        return len(words) <= 4

    if len(words) >= 2 and " ".join(words[:2]) in GREETING_WORDS:
        return len(words) <= 5

    return False


def is_thanks(message: str) -> bool:
    """Check whether the message is a thank-you message."""
    return any(message.startswith(phrase) for phrase in THANKS_PHRASES)


def is_goodbye(message: str) -> bool:
    """Check whether the message is a goodbye message."""
    return any(message.startswith(phrase) for phrase in GOODBYE_PHRASES)


def is_history_query(message: str) -> bool:
    """Check whether the message asks about previous conversation messages."""
    return any(keyword in message for keyword in HISTORY_KEYWORDS)


def is_deterministic_query(message: str) -> bool:
    """Check whether the message can be safely handled without an LLM."""
    normalized_message = normalize_message(message)

    return (
        is_greeting(normalized_message)
        or is_thanks(normalized_message)
        or is_goodbye(normalized_message)
        or is_history_query(normalized_message)
    )