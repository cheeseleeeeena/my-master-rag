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
import csv  # for CSV export


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


def get_sections(
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


def record_f1_categories(gold, predicted) -> List[Dict[str, Union[str, float]]]:
    """
    For each question, compute the maximum token-level F1 score between the predicted answer
    and all gold reference answers. Categorize the score into three buckets:
      - "below_0.5" for F1 scores less than 0.5,
      - "0.5_to_0.8" for F1 scores between 0.5 and 0.8 (inclusive), and
      - "over_0.8" for F1 scores above 0.8.
    Also record the original question, the gold answer, the predicted answer,
    the gold evidences, the predicted evidences, the gold answer type, and an additional
    "evidence-F1" score that indicates the evidence F1 for the question.

    When exporting to CSV, each question is written as a single row. The gold evidences
    are written in columns "gold_evidence_1" to "gold_evidence_6", and the predicted evidences
    are written in columns "predicted_evidence_1" to "predicted_evidence_3".
    """
    # Load test questions to get the original question text.
    questions_file: Path = Path("qasper/test_questions.json")
    with open(questions_file, "r") as file:
        test_questions: Dict[str, Dict] = json.load(file)

    records = []
    for question_id, gold_references in gold.items():
        if question_id in predicted:
            pred_answer = predicted[question_id]["answer"]
            pred_evidences = predicted[question_id]["evidence"]
        else:
            pred_answer = ""
            pred_evidences = []

        best_f1 = 0.0
        best_gold_answer = ""
        best_gold_evidences = []
        best_gold_answer_type = ""
        # Choose the gold reference that gives the best token-level F1.
        for reference in gold_references:
            current_f1 = token_f1_score(pred_answer, reference["answer"])
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_gold_answer = reference["answer"]
                best_gold_evidences = reference["evidence"]
                best_gold_answer_type = reference["type"]

        # Categorize based on F1 thresholds.
        if best_f1 < 0.5:
            category = "below_0.5"
        elif best_f1 <= 0.8:
            category = "0.5_to_0.8"
        else:
            category = "over_0.8"

        # Compute the evidence F1 score for the question as the maximum over all gold references.
        evidence_f1s = [
            paragraph_f1_score(pred_evidences, reference["evidence"])
            for reference in gold_references
        ]
        best_evidence_f1 = max(evidence_f1s) if evidence_f1s else 0.0

        # Retrieve the original question text; if not found, default to an empty string.
        original_question = (
            test_questions[question_id]["question"]
            if question_id in test_questions
            else ""
        )

        # Build a dictionary for gold evidences with columns gold_evidence_1 ... gold_evidence_6.
        max_gold = 6
        gold_evidences_dict = {}
        for i in range(max_gold):
            key = f"gold_evidence_{i+1}"
            if i < len(best_gold_evidences):
                gold_evidences_dict[key] = best_gold_evidences[i]
            else:
                gold_evidences_dict[key] = ""

        # Build a dictionary for predicted evidences with columns predicted_evidence_1 ... predicted_evidence_3.
        max_pred = 3
        pred_evidences_dict = {}
        for i in range(max_pred):
            key = f"predicted_evidence_{i+1}"
            if i < len(pred_evidences):
                pred_evidences_dict[key] = pred_evidences[i]
            else:
                pred_evidences_dict[key] = ""

        record = {
            "question_id": question_id,
            "f1_score": best_f1,
            "evidence-F1": best_evidence_f1,
            "category": category,
            "question": original_question,
            "gold_answer": best_gold_answer,
            "predicted_answer": pred_answer,
            "gold_answer_type": best_gold_answer_type,
        }
        record.update(gold_evidences_dict)
        record.update(pred_evidences_dict)

        records.append(record)
    return records


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

    # Prepare results directory.
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
        # Save to JSONLINES
        error_file = results_dir / "low_recall_cases.jsonl"
        with open(error_file, "w+", encoding="utf-8") as f:
            for error in all_errors:
                f.write(json.dumps(error, ensure_ascii=False))
                f.write("\n")

    # --- New functionality: record F1 categories for case study ---
    f1_category_records = record_f1_categories(
        gold_answers_and_evidence, predicted_answers_and_evidence
    )
    csv_file = results_dir / "f1_categories.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question_id",
            "f1_score",
            "evidence-F1",
            "category",
            "question",
            "gold_answer",
            "predicted_answer",
            "gold_answer_type",
            "gold_evidence_1",
            "gold_evidence_2",
            "gold_evidence_3",
            "gold_evidence_4",
            "gold_evidence_5",
            "gold_evidence_6",
            "predicted_evidence_1",
            "predicted_evidence_2",
            "predicted_evidence_3",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in f1_category_records:
            writer.writerow(record)
    print(f"F1 category records saved to {csv_file}")
