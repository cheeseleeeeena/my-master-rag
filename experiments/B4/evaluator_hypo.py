from collections import Counter
import argparse
import string
import re
import json
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import csv


def normalize_answer(s):
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


# return precision and recall scores separately
def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        return 1.0, 1.0, 1.0  # f1, precision, recall
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0, 0.0, 0.0  # f1, precision, recall
    precision = num_same / len(prediction) if prediction else 0.0
    recall = num_same / len(ground_truth) if ground_truth else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall


def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
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


def evaluate(gold, predicted, annotators):
    question_ids = list(predicted.keys())
    assert len(question_ids) == len(annotators), "something is wrong with input lists!"

    answer_f1s = []
    evidence_f1s = []
    evidence_precisions = []
    evidence_recalls = []
    answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    score_details = {}

    for qid, annotator in zip(question_ids, annotators):
        gold_answer = gold[qid][annotator]["answer"]
        gold_evidence = gold[qid][annotator]["evidence"]
        answer_type = gold[qid][annotator]["type"]

        answer_f1 = token_f1_score(predicted[qid]["answer"], gold_answer)
        answer_f1s.append(answer_f1)
        answer_f1s_by_type[answer_type].append(answer_f1)

        evidence_f1, evidence_precision, evidence_recall = paragraph_f1_score(
            predicted[qid]["evidence"], gold_evidence
        )

        evidence_f1s.append(evidence_f1)
        evidence_precisions.append(evidence_precision)
        evidence_recalls.append(evidence_recall)

        score_details[qid] = {
            "annotator_idx": annotator,
            "answer_f1": answer_f1,
            "evidence_f1": evidence_f1,
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
        }
    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(answer_f1s),
        "Answer F1 by type": {
            key: mean(value) for key, value in answer_f1s_by_type.items()
        },
        "Evidence F1": mean(evidence_f1s),
        "Evidence Precision": mean(evidence_precisions),
        "Evidence Recall": mean(evidence_recalls),
        "Score Details": score_details,
    }


def get_detailed_analysis(
    gold_data: Dict, predicted: Dict, text_evidence_only: bool, annotators
) -> List[Dict]:
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, text_evidence_only)
    eval_results = evaluate(gold_answers_and_evidence, predicted, annotators)

    score_details = eval_results["Score Details"]
    # organize paper details
    qid_to_paper_details = {}
    for paper_id, paper_info in gold_data.items():
        for qa_info in paper_info["qas"]:
            qid_to_paper_details[qa_info["question_id"]] = {
                "paper_id": paper_id,
                "question": qa_info["question"],
                "paper_title": paper_info["title"],
            }
    detailed_analysis = []
    for qid, predictions in predicted.items():

        predicted_scores = score_details.get(
            qid,
            {
                "annotator_idx": None,
                "answer_f1": None,
                "evidence_f1": None,
                "evidence_precision": None,
                "evidence_recall": None,
            },
        )
        annotator_idx = predicted_scores["annotator_idx"]
        answer_f1 = predicted_scores["answer_f1"]
        evidence_f1 = predicted_scores["evidence_f1"]
        evidence_precision = predicted_scores["evidence_precision"]
        evidence_recall = predicted_scores["evidence_recall"]

        gold_answer = (
            gold_answers_and_evidence[qid][annotator_idx]["answer"]
            if annotator_idx is not None
            else None
        )
        answer_type = (
            gold_answers_and_evidence[qid][annotator_idx]["type"]
            if annotator_idx is not None
            else None
        )
        gold_paras = (
            gold_answers_and_evidence[qid][annotator_idx]["evidence"]
            if annotator_idx is not None
            else []
        )
        predicted_answer = predictions["answer"]
        predicted_paras = predictions["evidence"]
        paper_details = qid_to_paper_details.get(
            qid, {"paper_id": None, "question": None, "paper_title": None}
        )
        analysis_entry = {
            "qid": qid,
            "question": paper_details["question"],
            "from_paper": paper_details["paper_id"],
            "paper_title": paper_details["paper_title"],
            "token_f1": answer_f1,
            "evidence_f1": evidence_f1,
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "annotator_idx": annotator_idx,
            # get the first 3 paras only since the maximum is 52 is too much
            "gold_paras": gold_paras[:3],
            "full_gold_paras": gold_paras,
            "predicted_paras": predicted_paras,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "gold_answer_type": answer_type,
        }
        detailed_analysis.append(analysis_entry)
    return detailed_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text_evidence_only", action="store_true", help="Ignore non-text evidence"
    )

    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset",
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Subdirectory name under 'predictions' to read prediction file and under 'eval_results' to save outputs",
    )

    parser.add_argument(
        "--subset", type=str, required=True, help="the id of subset of hypothesis fixes"
    )

    args = parser.parse_args()

    # Create results directory and subdirectory
    results_dir = Path("eval_results") / args.settings / args.subset
    results_dir.mkdir(parents=True, exist_ok=True)

    # read CSV file to get subset questions
    with open(
        results_dir / f"{args.subset}.csv",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        data = list(reader)  # Read remaining rows

    ## Transpose rows to columns, and convert tuples to lists, and unpack into lists
    columns = list(zip(*data))
    columns = [list(col) for col in columns]
    _, annotator_ids = columns
    annotator_ids = list(map(int, annotator_ids))

    # Load gold data
    gold_data = json.load(open(args.gold))

    # Load predictions
    predicted_answers_and_evidence = {}
    prediction_file: Path = (
        "predictions" / Path(f"llama3-{args.subset}") / "test_predictions.jsonl"
    )
    for line in open(prediction_file):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": prediction_data["predicted_answer"],
            "evidence": prediction_data["predicted_evidence"],
        }

    # Get evaluation results
    gold_answers_and_evidence = get_answers_and_evidence(
        gold_data, args.text_evidence_only
    )
    evaluation_output = evaluate(
        gold_answers_and_evidence, predicted_answers_and_evidence, annotator_ids
    )

    # Get detailed analysis
    detailed_analysis = get_detailed_analysis(
        gold_data,
        predicted_answers_and_evidence,
        args.text_evidence_only,
        annotator_ids,
    )

    # Save evaluation results as JSON
    eval_file = results_dir / "evaluation_scores.json"
    with open(eval_file, "w") as f:
        json.dump(evaluation_output, f, indent=2)
    print(f"Evaluation Results saved to {eval_file}")

    # Determine max number of paragraphs for gold and predicted
    gold_paras = (
        max(len(entry["gold_paras"]) for entry in detailed_analysis)
        if detailed_analysis
        else 0
    )
    pred_paras = (
        max(len(entry["predicted_paras"]) for entry in detailed_analysis)
        if detailed_analysis
        else 0
    )

    # Prepare data for CSV with separate columns for each paragraph and section
    csv_data = []

    for entry in detailed_analysis:
        row = {
            "qid": entry["qid"],
            "question": entry["question"],
            "paper_id": entry["from_paper"],
            "paper_title": entry["paper_title"],
            "annotator_idx": entry["annotator_idx"],
            "ans_f1": entry["token_f1"],
            "ev_f1": entry["evidence_f1"],
            "ev_precision": entry["evidence_precision"],
            "ev_recall": entry["evidence_recall"],
            "ans_type": entry["gold_answer_type"],
            "gold_answer": entry["gold_answer"],
            "predicted_answer": entry["predicted_answer"],
        }
        # Add gold paragraphs and sections as separate columns
        for i in range(gold_paras):
            row[f"gold_para_{i+1}"] = (
                entry["gold_paras"][i] if i < len(entry["gold_paras"]) else ""
            )
        # Add predicted paragraphs and sections as separate columns
        for i in range(pred_paras):
            row[f"predicted_para_{i+1}"] = (
                entry["predicted_paras"][i] if i < len(entry["predicted_paras"]) else ""
            )
        csv_data.append(row)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_data)
    analysis_file = results_dir / "fixed_analysis.csv"
    df.to_csv(analysis_file, index=False)
    print(f"Detailed Analysis saved to {analysis_file}")
