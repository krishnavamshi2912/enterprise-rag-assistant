"""Entry point for building and running the Telecom BSS knowledge assistant."""

from app.retrieval.pipeline import ask, build_teleco_assistant
from app.logger import get_logger

logger = get_logger(__name__)

def main():
    """Build the assistant, process a sample question, and display the response."""
    try:
        logger.info("Starting Telecom BSS knowledge assistant")
        agent = build_teleco_assistant()

        logger.info("Telecom BSS knowledge assistant is ready")
        question = "What is billing?"

        logger.info("Sending sample question to assistant: %s", question)
        answer = ask(agent, question)

        logger.info("Sample question processed successfully")
        
    except Exception:
        logger.exception("Application execution failed")
        raise

if __name__ == "__main__":
    main()
