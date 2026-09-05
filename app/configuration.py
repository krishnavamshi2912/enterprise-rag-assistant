import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


def get_required_env(name: str) -> str:
    """Get a required environment variable and raise an error if missing."""
    value = os.getenv(name)

    if not value:
        raise ValueError(f"{name} is not set in the environment variables.")

    return value


class Configuration:
    # Groq LLM configuration
    GROQ_API_KEY = get_required_env("GROQ_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-120b"

    # Fallback LLM configuration
    GROQ_FALL_BACK_API_KEY = get_required_env("GROQ_FALL_BACK_API_KEY")
    GROQ_FALL_BACK_MODEL = "openai/gpt-oss-20b"

    # Gemini embedding configuration
    GEMINI_API_KEY = get_required_env("GEMINI_API_KEY")

    # Qdrant vector database configuration
    QDRANT_URL = get_required_env("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = get_required_env("QDRANT_API_KEY")
    QDRANT_COLLECTION = "telecom-bss-rag-assistant"


configuration = Configuration()