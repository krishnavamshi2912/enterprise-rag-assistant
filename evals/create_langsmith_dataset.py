import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


DATASET_PATH = Path(__file__).parent / "dataset.json"

DATASET_NAME = "telecom-bss-rag-evaluation"

EVALUATION_CASES = [
    "bss_006",
    "bss_016",
    "bss_028",
    "bss_029",
    "bss_031",
]


def load_selected_cases() -> list:
    """Load the selected evaluation cases from the local dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return [
        case
        for case in dataset
        if case["id"] in EVALUATION_CASES
    ]


def build_examples(cases: list) -> list:
    """Convert local evaluation cases into LangSmith examples."""
    examples = []

    for case in cases:
        if "question" in case:
            inputs = {
                "question": case["question"]
            }
        else:
            inputs = {
                "conversation": case["conversation"]
            }

        outputs = {
            "expected_route": case["expected_route"],
            "expected_answer": case.get("expected_answer"),
            "relevant_sections": case.get("relevant_sections", []),
        }

        examples.append({
            "inputs": inputs,
            "outputs": outputs,
            "metadata": {
                "case_id": case["id"],
            },
        })

    return examples


def main() -> None:
    """Create the LangSmith dataset and add selected evaluation cases."""
    client = Client()

    cases = load_selected_cases()

    print(
        f"Loaded {len(cases)} cases: "
        f"{', '.join(case['id'] for case in cases)}"
    )

    try:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Evaluation dataset for the Telecom BSS RAG assistant. "
                "Contains selected smoke-test cases from evals/dataset.json."
            ),
        )

        print(f"Created LangSmith dataset: {dataset.name}")

    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(
                f"Dataset '{DATASET_NAME}' already exists."
            )

            dataset = next(
                client.list_datasets(
                    dataset_name=DATASET_NAME
                )
            )
        else:
            raise

    examples = build_examples(cases)

    response = client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    print(
        f"Added {len(examples)} examples to LangSmith."
    )
    print(f"Dataset ID: {dataset.id}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()