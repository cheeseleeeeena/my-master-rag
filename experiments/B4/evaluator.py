"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

from collections import Counter
import argparse
import string
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Union
import numpy as np


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_recall_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    return num_same / len(ground_truth)


def get_answers_and_evidence(
    data, text_evidence_only
) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references: List[Dict[str, Union[str, List[str]]]] = []
            # process all answers from different annotators
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append(
                        {"answer": "Unanswerable", "evidence": [], "type": "none"}
                    )
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(
                            f"Annotation {answer_info['annotation_id']} does not contain an answer"
                        )
                    if text_evidence_only:
                        evidence = [
                            text
                            for text in answer_info["evidence"]
                            if "FLOAT SELECTED" not in text
                        ]
                    else:
                        evidence = answer_info["evidence"]
                    references.append(
                        {"answer": answer, "evidence": evidence, "type": answer_type}
                    )
            answers_and_evidence[question_id] = references
    return answers_and_evidence


def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_evidence_recalls = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue
        answer_f1s_and_types = [
            (
                token_f1_score(predicted[question_id]["answer"], reference["answer"]),
                reference["type"],
            )
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(
            answer_f1s_and_types, key=lambda x: x[0], reverse=True
        )[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        evidence_f1s = [
            paragraph_f1_score(
                predicted[question_id]["evidence"], reference["evidence"]
            )
            for reference in gold[question_id]
        ]
        max_evidence_f1s.append(max(evidence_f1s))
        evidence_recalls = [
            paragraph_recall_score(
                predicted[question_id]["evidence"], reference["evidence"]
            )
            for reference in gold[question_id]
        ]
        max_evidence_recalls.append(max(evidence_recalls))

    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {
            key: mean(value) for key, value in max_answer_f1s_by_type.items()
        },
        "Evidence F1": mean(max_evidence_f1s),
        "Evidence Recall": mean(max_evidence_recalls),
        "Missing predictions": num_missing_predictions,
    }


def old_get_sections(
    predicted_paras: List[str], paper: Dict[str, Dict], paper_id: str
) -> List[str]:
    all_paras: List[str] = [para_obj["text"] for para_obj in paper[paper_id].values()]
    all_sections: List[str] = [
        para_obj["section_name"] for para_obj in paper[paper_id].values()
    ]
    predicted_sections: Set[str] = set()
    for predicted_para in predicted_paras:
        if predicted_para in all_paras:
            predicted_sections.add(all_sections[all_paras.index(predicted_para)])
    return list(predicted_sections)


def get_sections(
    predicted_paras: List[str], paper: Dict[str, Dict], paper_id: str
) -> List[str]:
    """
    Identify the originating section for each paragraph in predicted_paras.
    The output list maintains the same length as predicted_paras, including duplicates.

    Args:
        predicted_paras: List of predicted paragraphs to locate in the paper.
        paper: Dictionary with structure {paper_id: {'section_name': [str], 'text': [[str]]}}.
        paper_id: The ID of the paper to search within.

    Returns:
        List of section names corresponding to each predicted paragraph.
    """
    # Extract section names and text from the paper
    paper_data = paper[paper_id]
    section_names = paper_data["section_name"]  # List of section names
    section_texts = paper_data["text"]  # List of lists of paragraphs

    # Create a mapping of paragraphs to section names for O(1) lookups
    para_to_section: Dict[str, str] = {}
    for section_name, paragraphs in zip(section_names, section_texts):
        for para in paragraphs:
            # Map each paragraph to its section (last occurrence if duplicated in paper)
            para_to_section[para] = section_name

    # Find sections for predicted paragraphs, preserving order and length
    predicted_sections: List[str] = []
    for predicted_para in predicted_paras:
        # Append the section if found, otherwise append None (or an empty string, depending on preference)
        predicted_sections.append(para_to_section.get(predicted_para, None))

    return predicted_sections


def record_errors(gold, predicted) -> Dict[str, Dict[str, str]]:
    paper_file: Path = Path("qasper/test_papers.json")
    questions_file: Path = Path("qasper/test_questions.json")

    # get all paper contents from test set
    with open(paper_file, "r") as file:
        test_papers: Dict[str, Dict] = json.load(file)

    # get all questions from test set
    with open(questions_file, "r") as file:
        test_questions: Dict[str, Dict] = json.load(file)

    bad_evidences = []
    for question_id, gold_references in gold.items():
        paper_id: str = test_questions[question_id]["from_paper"]
        predicted_paras: List[str] = predicted[question_id]["evidence"]

        evidence_recalls = [
            paragraph_recall_score(predicted_paras, reference["evidence"])
            for reference in gold_references
        ]
        max_recall: float = max(evidence_recalls)
        # inspect only the reference answer where the model performs best
        best_ref_paras: List[str] = gold_references[np.argmax(evidence_recalls)][
            "evidence"
        ]

        if max_recall < 0.5:
            predicted_sections: List[str] = get_sections(
                predicted_paras, test_papers, paper_id
            )
            gold_sections: List[str] = get_sections(
                best_ref_paras, test_papers, paper_id
            )
            bad_evidences.append(
                {
                    "qid": question_id,
                    "question": test_questions[question_id]["question"],
                    "from_paper": paper_id,
                    "gold": best_ref_paras,
                    "gold_section": gold_sections,
                    "predicted": predicted_paras,
                    "predicted_section": predicted_sections,
                }
            )
    return bad_evidences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--predictions",
    #     type=str,
    #     required=True,
    #     help="""JSON lines file with each line in format:
    #             {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}""",
    # )

    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="""retriever-reader-mode-topk""",
    )

    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset",
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1",
    )

    parser.add_argument(
        "--rag",
        action="store_true",
        help="If set, the evaluator will record cases with low recall score.",
    )

    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(
        gold_data, args.text_evidence_only
    )

    # prepare results dir
    # experiment_settings: str = f"{args.retriever}-{args.reader}-{args.mode}-top{args.topk}"
    results_dir: Path = Path(f"results/{args.settings}")
    results_dir.mkdir(parents=True, exist_ok=True)

    predicted_answers_and_evidence = {}
    prediction_file: Path = results_dir / "test_predictions.jsonl"
    for line in open(prediction_file):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": prediction_data["predicted_answer"],
            "evidence": prediction_data["predicted_evidence"],
        }
    evaluation_output = evaluate(
        gold_answers_and_evidence, predicted_answers_and_evidence
    )

    result_file: Path = results_dir / f"{args.settings}.json"
    with open(result_file, "w+", encoding="utf-8") as f:
        f.write(json.dumps(evaluation_output, indent=2, ensure_ascii=False))

    if args.rag:
        all_errors: List[Dict] = record_errors(
            gold_answers_and_evidence, predicted_answers_and_evidence
        )
        print(f"Total low recall cases: {len(all_errors)}")
        # save to JSONLINES
        error_file = results_dir / "low_recall_cases.jsonl"
        with open(error_file, "w+", encoding="utf-8") as f:
            for error in all_errors:
                f.write(json.dumps(error, ensure_ascii=False))
                f.write("\n")
