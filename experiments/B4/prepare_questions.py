from datasets import load_dataset
from pathlib import Path
import json

# from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer
# import torch
import re

# from vllm import LLM


if __name__ == "__main__":
    raw_dataset = load_dataset(
        "allenai/qasper", cache_dir="/workspace/P76125041/.cache/huggingface/"
    )

    # Create a Path object for the output directory
    output_dir = Path(f"qasper/")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw dataset format
    # {
    #     "id": "Paper ID (string)",
    #     "qas": {
    #         "answers": [
    #             # only shows question1's answer
    #             {
    #                 "answer": [
    #                     # gold answer 1
    #                     {
    #                         "unanswerable": False,
    #                         "extractive_spans": [
    #                             "q1_answer1_extractive_span1",
    #                             "q1_answer1_extractive_span2",
    #                         ],
    #                         "yes_no": False,
    #                         "free_form_answer": "q1_answer1",
    #                         "evidence": [
    #                             "q1_answer1_evidence1",
    #                             "q1_answer1_evidence2",
    #                         ],
    #                         "highlighted_evidence": [
    #                             "q1_answer1_highlighted_evidence1",
    #                             "q1_answer1_highlighted_evidence2",
    #                         ],
    #                     },
    #                     # gold answer 2
    #                     {
    #                         "unanswerable": False,
    #                         "extractive_spans": [
    #                             "q1_answer2_extractive_span1",
    #                             "q1_answer2_extractive_span2",
    #                         ],
    #                         "yes_no": False,
    #                         "free_form_answer": "q1_answer2",
    #                         "evidence": [
    #                             "q1_answer2_evidence1",
    #                             "q1_answer2_evidence2",
    #                         ],
    #                         "highlighted_evidence": [
    #                             "q1_answer2_highlighted_evidence1",
    #                             "q1_answer2_highlighted_evidence2",
    #                         ],
    #                     },
    #                 ]
    #             },
    #             {"answer": "question2's answer"},
    #         ],
    #         "question": ["question1", "question2"],
    #         "question_id": ["question1_id", "question2_id"],
    #     },
    # }

    # Process dataset
    processed_questions = {}

    for paper in raw_dataset["test"]:

        paper_id = paper["id"]
        questions = paper["qas"]["question"]
        question_ids = paper["qas"]["question_id"]
        answers = paper["qas"]["answers"]

        for idx, question in enumerate(questions):
            question_id = question_ids[idx]
            answer_group = answers[idx]["answer"]

            processed_questions[question_id] = {
                "question": question,
                "from_paper": paper_id,
                "answers": answer_group,
            }

    # Save processed data to JSON
    output_file = output_dir / "processed_questions.json"
    with open(output_file, "w") as f:
        json.dump(processed_questions, f, indent=4)

    print(f"Processed data saved to {output_file}")
