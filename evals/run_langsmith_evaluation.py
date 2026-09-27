from langsmith import Client

from app.retrieval.llm import get_evaluator_llm
from app.retrieval.pipeline import build_teleco_assistant

DATASET_NAME = "telecom-bss-rag-evaluation"

graph = None
judge_llm = None


def get_graph():
    """Create the LangGraph once and reuse it."""
    global graph
    if graph is None:
        graph = build_teleco_assistant()
    return graph


def get_judge_llm():
    """Create the evaluator LLM once and reuse it."""
    global judge_llm
    if judge_llm is None:
        judge_llm = get_evaluator_llm()
    return judge_llm


def normalize_section(section: str) -> str:
    """Normalize section names for comparison."""
    section = section.strip().upper()
    if ". " in section:
        section = section.split(". ", 1)[1]
    return section


def get_question(inputs: dict) -> str:
    """Extract the user question from a dataset example."""
    if "question" in inputs:
        return inputs["question"]

    conversation = inputs.get("conversation", [])
    for message in reversed(conversation):
        if message.get("role") == "user":
            return message.get("content", "")

    return ""


def target(inputs: dict) -> dict:
    """Run one LangSmith dataset example through the LangGraph."""
    if "conversation" in inputs:
        messages = inputs["conversation"]
    else:
        messages = [{"role": "user", "content": inputs["question"]}]

    response = get_graph().invoke({
        "messages": messages,
        "route": "",
        "retrieval_query": "",
        "context": "",
        "answer": "",
        "retrieved_documents": [],
        "blocked": False,
    })

    route = "BLOCKED" if response["blocked"] else response["route"].upper()

    retrieved_sections = [
        normalize_section(document.metadata.get("section", "unknown"))
        for document in response["retrieved_documents"]
    ]

    return {
        "route": route,
        "answer": response["answer"],
        "context": response["context"],
        "retrieved_sections": retrieved_sections,
    }


def route_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Check whether the system selected the expected route."""
    expected_route = reference_outputs.get("expected_route", "").upper()
    actual_route = outputs.get("route", "").upper()

    return {
        "key": "route_correctness",
        "score": 1.0 if actual_route == expected_route else 0.0,
        "comment": f"Expected: {expected_route} | Actual: {actual_route}",
    }


def guardrail_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Check whether guardrail cases were correctly blocked."""
    expected_route = reference_outputs.get("expected_route", "").upper()

    if expected_route != "BLOCKED":
        return {
            "key": "guardrail_correctness",
            "value": None,
            "comment": "Not a guardrail evaluation case.",
        }

    actual_route = outputs.get("route", "").upper()

    return {
        "key": "guardrail_correctness",
        "score": 1.0 if actual_route == "BLOCKED" else 0.0,
        "comment": f"Expected: BLOCKED | Actual: {actual_route}",
    }


def precision_at_k(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    k: int = 4,
) -> dict:
    """Calculate Precision@K for retrieval-oriented cases."""
    relevant_sections = {
        normalize_section(section)
        for section in reference_outputs.get("relevant_sections", [])
    }

    if not relevant_sections:
        return {
            "key": f"precision_at_{k}",
            "value": None,
            "comment": "Retrieval metric not applicable.",
        }

    retrieved_sections = outputs.get("retrieved_sections", [])[:k]

    if not retrieved_sections:
        return {
            "key": f"precision_at_{k}",
            "score": 0.0,
            "comment": "No documents were retrieved.",
        }

    relevant_retrieved = sum(
        section in relevant_sections
        for section in retrieved_sections
    )

    precision = relevant_retrieved / len(retrieved_sections)

    return {
        "key": f"precision_at_{k}",
        "score": precision,
        "comment": (
            f"Relevant retrieved: {relevant_retrieved} | "
            f"Retrieved: {len(retrieved_sections)}"
        ),
    }


def recall_at_k(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    k: int = 4,
) -> dict:
    """Calculate Recall@K for retrieval-oriented cases."""
    relevant_sections = {
        normalize_section(section)
        for section in reference_outputs.get("relevant_sections", [])
    }

    if not relevant_sections:
        return {
            "key": f"recall_at_{k}",
            "value": None,
            "comment": "Retrieval metric not applicable.",
        }

    retrieved_sections = set(
        outputs.get("retrieved_sections", [])[:k]
    )

    relevant_retrieved = relevant_sections.intersection(retrieved_sections)
    recall = len(relevant_retrieved) / len(relevant_sections)

    return {
        "key": f"recall_at_{k}",
        "score": recall,
        "comment": (
            f"Relevant sections found: {len(relevant_retrieved)} | "
            f"Expected: {len(relevant_sections)}"
        ),
    }


def mean_reciprocal_rank(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    k: int = 4,
) -> dict:
    """Calculate Reciprocal Rank for the first relevant result."""
    relevant_sections = {
        normalize_section(section)
        for section in reference_outputs.get("relevant_sections", [])
    }

    if not relevant_sections:
        return {
            "key": "mrr",
            "value": None,
            "comment": "Retrieval metric not applicable.",
        }

    retrieved_sections = outputs.get("retrieved_sections", [])[:k]

    for rank, section in enumerate(retrieved_sections, start=1):
        if section in relevant_sections:
            reciprocal_rank = 1 / rank
            return {
                "key": "mrr",
                "score": reciprocal_rank,
                "comment": (
                    f"First relevant result rank: {rank} | "
                    f"Reciprocal rank: {reciprocal_rank:.2f}"
                ),
            }

    return {
        "key": "mrr",
        "score": 0.0,
        "comment": "No relevant result found in top K.",
    }


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate whether the answer is supported by retrieved context."""
    if outputs.get("route") != "TECHNICAL":
        return {
            "key": "groundedness",
            "value": None,
            "comment": "Groundedness not applicable.",
        }

    context = outputs.get("context", "")
    answer = outputs.get("answer", "")
    question = get_question(inputs)

    if not context:
        return {
            "key": "groundedness",
            "score": 0.0,
            "comment": "No retrieved context available.",
        }

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Your task is to evaluate whether the generated answer is grounded
strictly in the retrieved context.

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Evaluation criteria:

1. Give 1.0 if the answer is fully supported by the retrieved context.
2. Give 0.5 if the answer is only partially supported.
3. Give 0.0 if the answer contains unsupported claims or contradicts
   the retrieved context.

Do not judge whether the answer sounds good.
Only judge whether its claims are supported by the retrieved context.

Return ONLY a number:
1.0, 0.5, or 0.0
"""

    response = get_judge_llm().invoke(prompt)

    try:
        score = float(response.content.strip())
    except (ValueError, TypeError):
        score = 0.0

    score = max(0.0, min(1.0, score))

    return {
        "key": "groundedness",
        "score": score,
        "comment": "Evaluated by the separate LLM evaluator through Portkey.",
    }


def answer_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Evaluate whether the generated answer is correct."""
    expected_answer = reference_outputs.get("expected_answer")

    if not expected_answer:
        return {
            "key": "answer_correctness",
            "value": None,
            "comment": "No expected answer available.",
        }

    question = get_question(inputs)
    answer = outputs.get("answer", "")

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate whether the generated answer correctly answers the question.

Question:
{question}

Expected answer:
{expected_answer}

Generated answer:
{answer}

Evaluation criteria:

1. Give 1.0 if the generated answer is fully correct and agrees
   with the expected answer.
2. Give 0.5 if the answer is partially correct but misses or
   incorrectly handles an important detail.
3. Give 0.0 if the answer is incorrect.

The generated answer may contain additional correct explanation.
Do not require exact wording.

Return ONLY a number:
1.0, 0.5, or 0.0
"""

    response = get_judge_llm().invoke(prompt)

    try:
        score = float(response.content.strip())
    except (ValueError, TypeError):
        score = 0.0

    score = max(0.0, min(1.0, score))

    return {
        "key": "answer_correctness",
        "score": score,
        "comment": "Evaluated by the separate LLM evaluator through Portkey.",
    }


def answer_relevance(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Evaluate whether the answer directly addresses the question."""
    question = get_question(inputs)
    answer = outputs.get("answer", "")

    if not question:
        return {
            "key": "answer_relevance",
            "value": None,
            "comment": "No question available.",
        }

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate whether the generated answer directly addresses the user's
question.

Question:
{question}

Generated answer:
{answer}

Evaluation criteria:

1. Give 1.0 if the answer directly and sufficiently addresses
   the question.
2. Give 0.5 if the answer addresses the question only partially
   or contains some unnecessary information.
3. Give 0.0 if the answer does not address the question.

Do not judge factual correctness here.
Only evaluate relevance to the question.

Return ONLY a number:
1.0, 0.5, or 0.0
"""

    response = get_judge_llm().invoke(prompt)

    try:
        score = float(response.content.strip())
    except (ValueError, TypeError):
        score = 0.0

    score = max(0.0, min(1.0, score))

    return {
        "key": "answer_relevance",
        "score": score,
        "comment": "Evaluated by the separate LLM evaluator through Portkey.",
    }


def main():
    """Run the LangGraph against the LangSmith evaluation dataset."""
    client = Client()

    print(f"Running LangSmith dataset: {DATASET_NAME}")

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        experiment_prefix="telecom-bss-rag",
        description="Smoke-test evaluation of the Telecom BSS RAG assistant.",
        evaluators=[
            route_correctness,
            guardrail_correctness,
            precision_at_k,
            recall_at_k,
            mean_reciprocal_rank,
            groundedness,
            answer_correctness,
            answer_relevance,
        ],
        max_concurrency=1,
        blocking=True,
    )

    print("LangSmith evaluation completed.")
    print(results)


if __name__ == "__main__":
    main()