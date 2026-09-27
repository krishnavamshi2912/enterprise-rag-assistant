# RAG Application with LangGraph-Orchestrated Agent Workflows

A RAG application built using LangChain and LangGraph to answer questions from a Telecom BSS knowledge base.

The application uses query routing, vector search, FlashRank reranking, guardrails, and LLM-based response generation.

## Features

- LangGraph-based query routing
- RAG pipeline with Qdrant vector search
- Section-aware document chunking
- FlashRank reranking 
- Context-grounded responses
- Prompt injection protection
- Portkey for LLM routing
- LangSmith for evaluation and tracing
- Streamlit UI

## Flow

## Flow

```text
User Query
    ↓
Input Guardrail
    ├── blocked ─────────────────────────→ End
    │
    └── allowed
          ↓
       Planner
       ├── direct ───────────────────────→ End
       │
       ├── technical
       │      ↓
       │   Retriever
       │      ↓
       │   Responder
       │      ↓
       │ Output Guardrail
       │      ↓
       │   allowed ─────────────────────→ End
       │
       └── out_of_domain
              ↓
       Domain Rejection
              ↓
             End
```

For technical queries, the application retrieves relevant chunks from Qdrant and reranks them using FlashRank before generating the response.

## Tech Stack

- Python
- LangChain
- LangGraph
- Qdrant
- Jina Embeddings
- FlashRank
- Groq
- Portkey
- Streamlit
- FastAPI
- LangSmith
- Logging

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/krishnavamshi2912/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` and add the required API keys for Qdrant, Groq, Jina, Portkey, and LangSmith.

### 5. Run the backend

```bash
python -m uvicorn app.main:app --reload
```

### 6. Run the frontend

Open another terminal and run:

```bash
streamlit run app/ui/streamlit_app.py
```


## Author

**Krishna Vamshi**
GitHub: [krishnavamshi2912](https://github.com/krishnavamshi2912)