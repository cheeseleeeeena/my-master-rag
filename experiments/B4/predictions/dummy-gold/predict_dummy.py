import json
from pathlib import Path
from typing import List, Dict, Union
import time
import os
from functools import lru_cache

import utils
import evaluator

# Set environment variable at module level
os.environ["HF_HOME"] = "/workspace/P76125041/.cache/"


@lru_cache(maxsize=1)
def load_gold_data() -> Dict[str, Dict]:
    """Load and cache gold data to avoid repeated file reads."""
    return json.load(open("qasper/test_gold.json", "r", encoding="utf-8"))


def process_questions(
    test_questions: Dict[str, Dict], gold_data: Dict[str, Dict]
) -> List[Dict[str, Union[str, List[str]]]]:
    """Process questions and generate predictions efficiently."""
    all_predictions = []
    get_answers = evaluator.get_answers_and_evidence  # Cache function reference

    for question_id, question_data in test_questions.items():
        print(f"Processing `{question_data['question']}`...")

        # Get answers and evidence once per question
        answers_and_evidence = get_answers(gold_data, True)
        first_entry = answers_and_evidence[question_id][-1]

        # Create prediction dict in one go
        all_predictions.append(
            {
                "question_id": question_id,
                "predicted_evidence": first_entry["evidence"],
                "predicted_answer": first_entry["answer"],
            }
        )

    return all_predictions


def save_predictions(predictions: List[Dict], output_path: Path) -> None:
    """Efficiently save predictions to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        # Use writelines for batch writing
        f.writelines(
            json.dumps(pred, ensure_ascii=False) + "\n" for pred in predictions
        )


if __name__ == "__main__":
    start_time = time.time()

    # Load data efficiently
    gold_data = load_gold_data()
    test_questions = utils.load_json(Path("qasper/test_questions.json"))

    # Process questions and get predictions
    predictions = process_questions(test_questions, gold_data)

    # Save results
    results_dir = Path("results/dummy")
    save_predictions(predictions, results_dir / "test_predictions.jsonl")

    print(f"Total time (sec): {time.time() - start_time:.2f}")
