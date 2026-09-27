# Evaluation Dataset

This directory contains the evaluation dataset and LangSmith evaluation workflow used to assess the Telecom BSS RAG assistant.

## Evaluation Categories

The evaluation dataset contains 35 test cases covering:

- Telecom BSS technical questions
- Calculation-based questions
- Concept comparison questions
- Multi-turn follow-up questions
- Conversation history queries
- Direct conversational queries
- Out-of-domain questions
- Prompt injection and safety cases

## Evaluation Approach

The project uses **LangSmith** for evaluation and experiment tracking.

The evaluation workflow validates different parts of the RAG application.

### Routing

- **Route Correctness** — verifies whether the planner selects the expected route such as `DIRECT`, `CONVERSATIONAL`, or `TECHNICAL`.
- **Guardrail Correctness** — verifies whether prompt injection and other blocked cases are correctly prevented.

### Retrieval

The application retrieves **8 candidate chunks** from the vector store and applies **FlashRank reranking** using the `ms-marco-TinyBERT-L-2-v2` model.

FlashRank reranks the retrieved candidates based on their relevance to the user's query.

The evaluation measures the top 4 reranked results:

- **Precision@4** — proportion of the top 4 retrieved chunks that belong to the expected relevant sections.
- **Recall@4** — proportion of expected relevant sections found within the top 4 retrieved chunks.
- **MRR (Mean Reciprocal Rank)** — measures the ranking position of the first relevant result within the top 4 results.

### Answer Generation

The generated answers are evaluated using a separate evaluator LLM through the Portkey gateway:

- **Groundedness** — checks whether the generated answer is supported by the retrieved context.
- **Answer Correctness** — checks whether the generated answer correctly addresses the expected answer.
- **Answer Relevance** — checks whether the generated answer directly addresses the user's question.

## Evaluation Dataset

The complete local dataset contains **35 evaluation cases** in `dataset.json`.

Each case can contain:

- Case identifier
- User question or conversation history
- Expected route
- Expected answer, where applicable
- Relevant knowledge-base sections for retrieval evaluation

Multi-turn cases use conversation history instead of a single question field.

The expected answer is used as a semantic reference. Exact string matching is not required.

## Smoke Test

To avoid unnecessary model calls and rate-limit issues during development, the project currently runs a **5-case smoke test** selected from the full 35-case dataset.

The selected cases are configured in:

`create_langsmith_dataset.py`

The current smoke test covers:

- A direct conversational query
- A calculation-based technical query
- A prompt injection case
- A multi-turn technical follow-up
- A technical knowledge query

This provides a quick regression check across routing, guardrails, retrieval, and answer generation.

The full 35-case dataset remains available for broader evaluation when required.

## Evaluation Workflow

The evaluation workflow consists of two steps.

### 1. Create the LangSmith Dataset

Run:

```bash
python -m evals.create_langsmith_dataset
```

This reads the selected cases from `dataset.json` and creates the LangSmith dataset:

`telecom-bss-rag-evaluation`

### 2. Run the Evaluation

Run:

```bash
python -m evals.run_langsmith_evaluation
```

The evaluation executes the selected dataset cases through the LangGraph application and records the custom evaluation metrics in LangSmith.

## Evaluation Metrics

The current evaluation reports:

| Area | Metric |
|---|---|
| Routing | Route Correctness |
| Safety | Guardrail Correctness |
| Retrieval | Precision@4 |
| Retrieval | Recall@4 |
| Retrieval | MRR |
| Generation | Groundedness |
| Generation | Answer Correctness |
| Generation | Answer Relevance |

## Example Smoke-Test Results

A recent 5-case smoke test produced the following aggregate results:

| Metric | Score |
|---|---:|
| Route Correctness | 1.00 |
| Guardrail Correctness | 1.00 |
| Groundedness | 1.00 |
| Answer Correctness | 1.00 |
| Answer Relevance | 0.75 |
| Recall@4 | 1.00 |
| MRR | 1.00 |
| Precision@4 | 0.63 |

These results represent a **development-time smoke test**, not a statistically significant benchmark of production performance.

The evaluation also records execution information such as latency and token usage through LangSmith.

## Retrieval Evaluation Flow

The retrieval and evaluation flow can be summarized as:

```text
User Query
    ↓
Vector Retrieval
    ↓
8 Candidate Chunks
    ↓
FlashRank Reranker
(ms-marco-TinyBERT-L-2-v2)
    ↓
Reranked Results
    ↓
Top 4 Used for Evaluation
    ↓
Precision@4 / Recall@4 / MRR
```

The same reranked context is used by the RAG pipeline to generate the final answer.

## Files

```text
evals/
├── dataset.json
├── create_langsmith_dataset.py
├── run_langsmith_evaluation.py
└── README.md
```

- `dataset.json` — complete 35-case evaluation dataset
- `create_langsmith_dataset.py` — creates the selected LangSmith evaluation dataset
- `run_langsmith_evaluation.py` — runs the LangGraph application and custom LangSmith evaluators
- `README.md` — evaluation documentation