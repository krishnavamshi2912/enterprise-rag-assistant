"""FastAPI entry point for the Telecom BSS RAG assistant."""

from fastapi import FastAPI, Response
from pydantic import BaseModel

from app.retrieval.pipeline import ask, build_teleco_assistant
from app.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Telecom BSS RAG API")

graph = None


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    """Build the Telecom BSS assistant when the API starts."""
    global graph

    try:
        logger.info("Starting Telecom BSS RAG API")
        graph = build_teleco_assistant()
        logger.info("Telecom BSS RAG assistant is ready")
    except Exception:
        logger.exception("Failed to start Telecom BSS RAG API")
        raise


@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Telecom BSS RAG API is live."}


@app.get("/graph")
def get_graph():
    """Return the LangGraph workflow as a PNG image."""
    try:
        logger.info("Generating LangGraph workflow image")

        png_bytes = graph.get_graph().draw_mermaid_png()

        logger.info("LangGraph workflow image generated successfully")

        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        logger.exception("Failed to generate LangGraph workflow image")
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """Process a user question through the Telecom BSS LangGraph assistant."""
    try:
        logger.info("API query received | question=%s", request.question)

        answer = ask(graph, request.question)

        logger.info("API query processed successfully")

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception:
        logger.exception("Failed to process API query")

        return {
            "question": request.question,
            "answer": "I apologize, but I encountered an internal error while processing your request.",
            "status": "error"
        }